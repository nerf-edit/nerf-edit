### Generate Grounded-SAM Features

```
python grounded_sam_inference.py --config <path to grounding sam root>/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint <path to Grounding Dino checkpoint: groundingdino_swint_ogc.pth>   --sam_checkpoint <path to SAM checkpoint: sam_vit_h_4b8939.pth>   --input < path to color folder after preprocessing datasets > --output < path to save panoptic and text feature predictions >   --box_threshold 0.3   --text_threshold 0.25   --text_prompt <all scene labels seperated by comma> --device "cuda"
```