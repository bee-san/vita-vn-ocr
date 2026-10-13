#!/usr/bin/env bash
# The three study-2 test reports (eval/synth3_test_report.py over $S3/study2/test, after the matching result commit):
# FL10 (f79c45d / 8a02ab8), H15 (57da7a5), and the step-curve exports (02063b0). Reported only.
set -euo pipefail
S3=/local/home/skerraut/herd-vita-ocr/shared/artifacts/synth3; T=$S3/study2/test
cd "$(dirname "$0")/../../.."
P=eval/synth3_study2_protocol.json; SPEC=eval/synth3_study2_reported.json
.venv-ci/bin/python eval/synth3_test_report.py --protocol $P --dir $T --spec $SPEC --arms FL10_A_s2,FL10_C_mix,pre_FL10_A_s2,pre_FL10_C_mix \
  --ref REF_FL10_v020 --out eval/synth3_study2_FL10_test_20261012.json
.venv-ci/bin/python eval/synth3_test_report.py --protocol $P --dir $T --spec $SPEC --arms H15_A_s2,H15_C_mix,pre_H15_A_s2,pre_H15_C_mix \
  --ref REF_H15_v020 --out eval/synth3_study2_H15_test_20261013.json
.venv-ci/bin/python eval/synth3_test_report.py --protocol $P --dir $T --spec $SPEC --arms FL10_C_mix_32k_s1,FL10_C_mix_64k_s1,FL10_C_mix_128k_s1 \
  --ref REF_FL10_v020 --ref pre_FL10_C_mix_pre64k_seed1 --ref pre_FL10_C_mix_pre128k_seed1 --out eval/synth3_study2_FL10_step_test_20261013.json
