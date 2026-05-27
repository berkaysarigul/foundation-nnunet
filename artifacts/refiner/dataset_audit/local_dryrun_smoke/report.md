# PR-10B manifest validation report

- Generated at: 2026-05-27T04:43:06.959063+00:00
- Strict mode: False
- Overall verdict: **FAIL**
- Failed checks (2): ['test:test_split_value', 'cross:case_id_disjoint_train_test']

## train manifest
- Path: `artifacts\diagnostics\foundation_x_official_direct\teacher_head5_official224_export_imagesTr_dryrun\teacher_head5_prior_manifest_train.csv`
- Row count: 2
- Positives / negatives: 0 / 2
- Sampled image sizes: {'512x512': 2}
- Sampled label sizes: {'512x512': 2}
- Sampled prob sizes: {'224x224': 2}
- Sampled prob range: [0, 0]
- Sampled label unique values seen: [0]

### Checks
  - [PASS] schema_header: schema matches PR-10A 20-column contract
  - [PASS] train_row_count: row count not validated; got 2 (no --expected-train-rows supplied)
  - [PASS] train_case_id_unique: 0 duplicate case_id(s)
  - [PASS] train_split_value: observed splits=['train']; allowed=['train']
  - [PASS] train_provenance: expected {'state_key': 'teacher_model', 'preprocess_variant': 'official_siim_224', 'head_key': 'head_5'}
  - [PASS] train_leakage_imagesTs: no path may include the 'imagesTs/' segment in train manifest
  - [PASS] train_leakage_heldout_labelsTs: no path may include the 'heldout_labelsTs/' segment in train manifest
  - [PASS] train_image_paths_exist: 0 missing image_path
  - [PASS] train_label_paths_exist: 0 missing label_path
  - [PASS] train_probability_map_paths_exist: 0 missing probability_map_path
  - [PASS] train_label_binarity_sampled: sampled label pixel values must be a subset of {0, 1, 255}
  - [PASS] train_image_label_alignment_sampled: sampled image and label must share H x W
  - [PASS] train_prob_range_sampled: sampled prob uint8 range observed: [0, 0]
  - [WARN] train_prob_nondegenerate_sampled: sampled prob maps with min == max are usually all-zero placeholders
      first offenders: ['siim_000001', 'siim_000002']
  - [PASS] train_prob_size_uniform_sampled: sampled prob H x W values: {'224x224': 2}

## test manifest
- Path: `artifacts\diagnostics\foundation_x_official_direct\teacher_head5_official224_export_imagesTr_dryrun\teacher_head5_prior_manifest_train.csv`
- Row count: 2
- Positives / negatives: 0 / 2
- Sampled image sizes: {'512x512': 2}
- Sampled label sizes: {'512x512': 2}
- Sampled prob sizes: {'224x224': 2}
- Sampled prob range: [0, 0]
- Sampled label unique values seen: [0]

### Checks
  - [PASS] schema_header: schema matches PR-10A 20-column contract
  - [PASS] test_row_count: row count not validated; got 2 (no --expected-test-rows supplied)
  - [PASS] test_case_id_unique: 0 duplicate case_id(s)
  - [FAIL] test_split_value: observed splits=['train']; allowed=['eval', 'heldout', 'test', 'val']
      first offenders: ['siim_000001', 'siim_000002']
  - [PASS] test_provenance: expected {'state_key': 'teacher_model', 'preprocess_variant': 'official_siim_224', 'head_key': 'head_5'}
  - [PASS] test_leakage_imagesTr: --allow-imagesTr-in-test was set; check skipped
  - [PASS] test_image_paths_exist: 0 missing image_path
  - [PASS] test_label_paths_exist: 0 missing label_path
  - [PASS] test_probability_map_paths_exist: 0 missing probability_map_path
  - [PASS] test_label_binarity_sampled: sampled label pixel values must be a subset of {0, 1, 255}
  - [PASS] test_image_label_alignment_sampled: sampled image and label must share H x W
  - [PASS] test_prob_range_sampled: sampled prob uint8 range observed: [0, 0]
  - [WARN] test_prob_nondegenerate_sampled: sampled prob maps with min == max are usually all-zero placeholders
      first offenders: ['siim_000001', 'siim_000002']
  - [PASS] test_prob_size_uniform_sampled: sampled prob H x W values: {'224x224': 2}

## Cross-manifest
  - [FAIL] case_id_disjoint_train_test: 2 case_id(s) appear in both train and test manifests
      first offenders: ['siim_000001', 'siim_000002']
