"""Static checks for generated self-contained entrypoints."""
import ast,json,re
from pathlib import Path
ROOT=Path(__file__).parent
expected={'01_global_convnext':'convnext_small.in12k_ft_in1k_384','02_global_coatnet':'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k','03_plane_contrast':'convnext_small.in12k_ft_in1k_384','04_roi_specialist':'tf_efficientnet_b2.ns_jft_in1k','05_efficientnet_global':'tf_efficientnet_b2.ns_jft_in1k','06_metric_convnext':'convnext_tiny.in12k_ft_in1k'}
for stem,encoder in expected.items():
 p=ROOT/(stem+'.py');s=p.read_text();ast.parse(s);assert len(s)>500_000 and encoder in s;print(stem,'ok',len(s),encoder)
assert len(list(ROOT.glob('0*.py')))==6
