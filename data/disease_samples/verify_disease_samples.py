"""Exhaustive verification suite for clinical pathway token and parameter packages."""

import json
import os
import struct
import numpy as np
from safetensors.numpy import load_file

BASE_DIR = "." if os.path.exists("shared_vocab.json") else "disease_samples"


def cosine_sim(a, b):
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    return float(np.dot(a_f, b_f) / ((np.linalg.norm(a_f) * np.linalg.norm(b_f)) + 1e-8))


def verify_all():
    with open(os.path.join(BASE_DIR, "shared_vocab.json"), "r", encoding="utf-8") as f:
        shared_vocab = json.load(f)
    global_t2i = shared_vocab["token_to_id"]
    global_vocab_size = shared_vocab["vocab_size"]
    assert len(global_t2i) == global_vocab_size

    disease_dirs = sorted(
        d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))
    )
    assert len(disease_dirs) == 5, f"Expected 5 diseases, found {len(disease_dirs)}"

    global_id_owner = {}
    shared_token_vectors = {}
    summary_rows = []

    for d_name in disease_dirs:
        dp = os.path.join(BASE_DIR, d_name)
        tokens_dir = os.path.join(dp, "tokens")
        params_dir = os.path.join(dp, "parameters")

        # 1. File presence check
        required_files = [
            os.path.join(tokens_dir, "mapping_table.json"),
            os.path.join(tokens_dir, "token_array.json"),
            os.path.join(tokens_dir, "token_array.bin"),
            os.path.join(tokens_dir, "tokenizer.json"),
            os.path.join(params_dir, "weights.safetensors"),
            os.path.join(params_dir, "param_metadata.json"),
        ]
        for rf in required_files:
            assert os.path.exists(rf), f"Missing required file: {rf}"

        # 2. Load tokens & mapping
        with open(os.path.join(tokens_dir, "mapping_table.json"), "r", encoding="utf-8") as f:
            mp = json.load(f)
        with open(os.path.join(tokens_dir, "token_array.json"), "r", encoding="utf-8") as f:
            arr = json.load(f)
        with open(os.path.join(tokens_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            hf_tok = json.load(f)
        with open(os.path.join(tokens_dir, "token_array.bin"), "rb") as f:
            raw_bin = f.read()
        bin_ids = list(struct.unpack(f"<{len(raw_bin) // 2}H", raw_bin))

        # Verify HF tokenizer schema
        assert hf_tok["model"]["type"] == "WordLevel"
        assert hf_tok["model"]["vocab"] == global_t2i

        # Verify binary == json array
        tids = arr["token_ids"]
        assert bin_ids == tids, f"{d_name}: binary array mismatch"
        n_tok = len(tids)
        assert n_tok == arr["token_count"]
        assert len(arr["attention_mask"]) == n_tok
        assert len(arr["token_type_ids"]) == n_tok
        assert len(arr["requirement_mask"]) == n_tok

        # Verify mapping roundtrip & global uniqueness
        id2tok = mp["id_to_token"]
        tok2id = mp["token_to_id"]
        assert [id2tok[str(i)] for i in tids] == arr["token_sequence"], f"{d_name}: roundtrip mismatch"
        for tok, tid in tok2id.items():
            assert global_t2i[tok] == tid, f"{d_name}: global ID mismatch for {tok}"
            if tid in global_id_owner:
                assert global_id_owner[tid] == tok, f"ID collision at {tid}: {global_id_owner[tid]} vs {tok}"
            global_id_owner[tid] = tok

        # 3. Verify safetensors parameters
        sf_path = os.path.join(params_dir, "weights.safetensors")
        w = load_file(sf_path)
        with open(os.path.join(params_dir, "param_metadata.json"), "r", encoding="utf-8") as f:
            pmeta = json.load(f)

        expected_tensors = {
            "embedding.weight",
            "active_vocab_mask",
            "prefix_prompt.weight",
            "stage_transition.weight",
            "lora_q_proj.lora_A.weight",
            "lora_q_proj.lora_B.weight",
            "lora_v_proj.lora_A.weight",
            "lora_v_proj.lora_B.weight",
        }
        assert set(w.keys()) == expected_tensors, f"{d_name}: tensor keys mismatch"

        # Numerical finiteness check across all tensors
        for tname, tarr in w.items():
            if np.issubdtype(tarr.dtype, np.floating):
                assert np.all(np.isfinite(tarr)), f"{d_name}.{tname} contains NaN/Inf!"

        emb = w["embedding.weight"]
        mask = w["active_vocab_mask"]
        assert emb.shape == (global_vocab_size, 4096)
        assert emb.dtype == np.float16
        assert mask.shape == (global_vocab_size,)
        assert int(mask.sum()) == len(tok2id)

        # Direct indexing test (zero IndexError)
        seq_vecs = emb[tids]
        assert seq_vecs.shape == (n_tok, 4096)
        seq_norms = np.linalg.norm(seq_vecs.astype(np.float32), axis=1)
        assert np.all(seq_norms > 1.0), f"{d_name}: active token has zero/tiny norm"

        # Inactive rows must be zero
        inactive_norms = np.linalg.norm(emb[mask == 0].astype(np.float32), axis=1)
        assert np.allclose(inactive_norms, 0.0), f"{d_name}: inactive rows not zero"

        # Cross-disease bitwise vector consistency for shared tokens
        for tok, tid in tok2id.items():
            vec = emb[tid]
            if tok in shared_token_vectors:
                assert np.array_equal(shared_token_vectors[tok], vec), (
                    f"Vector mismatch for shared token {tok} in {d_name}"
                )
            else:
                shared_token_vectors[tok] = vec.copy()

        # Verify semantic clustering (intra-category sim > inter-category sim)
        lab_ids = [global_t2i[s["token"]] for s in arr["step_by_step_timeline"] if s["category"] == "lab"]
        med_ids = [global_t2i[s["token"]] for s in arr["step_by_step_timeline"] if s["category"] == "med"]
        intra_lab = cosine_sim(emb[lab_ids[0]], emb[lab_ids[1]])
        inter_lab_med = cosine_sim(emb[lab_ids[0]], emb[med_ids[0]])
        assert intra_lab > inter_lab_med + 0.25, (
            f"{d_name}: poor semantic clustering intra_lab={intra_lab:.3f}, inter={inter_lab_med:.3f}"
        )

        # Verify Markov transition matrix
        trans = w["stage_transition.weight"]
        assert trans.dtype == np.float32
        assert np.all(trans >= 0.0)
        row_sums = trans.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"{d_name}: transition row sums != 1.0"
        # Absorbing states (last two: Discharge & VARIATION_EXIT)
        assert np.isclose(trans[-1, -1], 1.0) and np.isclose(trans[-2, -2], 1.0)
        # No impossible skip from stage 0 directly to Discharge (-2)
        assert trans[0, -2] == 0.0, f"{d_name}: impossible D1 -> Discharge jump!"

        # Verify LoRA low-rank matrices
        qa, qb = w["lora_q_proj.lora_A.weight"], w["lora_q_proj.lora_B.weight"]
        assert qa.shape == (16, 4096) and qb.shape == (4096, 16)
        delta_w_norm = float(np.linalg.norm(qb.astype(np.float32) @ qa.astype(np.float32)))
        assert 0.0 < delta_w_norm < 5.0, f"{d_name}: abnormal LoRA deltaW norm {delta_w_norm}"

        summary_rows.append(
            {
                "disease": d_name,
                "seq_tokens": n_tok,
                "active_vocab": len(tok2id),
                "global_vocab": global_vocab_size,
                "bin_bytes": len(raw_bin),
                "sf_kb": round(os.path.getsize(sf_path) / 1024, 1),
                "total_params": pmeta["total_parameters"],
                "intra_lab_cos": round(intra_lab, 4),
                "inter_lab_med_cos": round(inter_lab_med, 4),
                "lora_dw_norm": round(delta_w_norm, 4),
            }
        )

    print("=== ALL VERIFICATION CHECKS PASSED (0 ERRORS, 0 COLLISIONS) ===")
    for row in summary_rows:
        print(
            f"[{row['disease']}] seq={row['seq_tokens']} | vocab={row['active_vocab']}/{row['global_vocab']} "
            f"| bin={row['bin_bytes']}B | safetensors={row['sf_kb']}KB ({row['total_params']} params) "
            f"| cos(lab,lab)={row['intra_lab_cos']} vs cos(lab,med)={row['inter_lab_med_cos']} "
            f"| lora_dW_norm={row['lora_dw_norm']}"
        )


if __name__ == "__main__":
    verify_all()
