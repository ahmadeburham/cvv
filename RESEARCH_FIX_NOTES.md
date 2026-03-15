# Research-based fixes applied

## Main code shortcomings addressed
- Fixed-ROI extraction was brittle on photographed cards. The updated pipeline keeps template-casting for card localization, but uses larger semantic blocks (`full_name`, `full_address`) and line segmentation inside those blocks before OCR.
- Validation depended too heavily on raw grayscale SSIM. The updated pipeline combines grayscale similarity with edge overlap on stable artwork patches.
- OCR startup and inference were inconsistent across environments. The updated pipeline uses PaddleOCR first, with document orientation classification, unwarping, and textline orientation enabled; it falls back to Tesseract if PaddleOCR is unavailable.
- Numeric fields were weakly validated. The updated pipeline validates Egyptian national ID format and infers birthday from the 14-digit ID when possible.
- Folder layout is now fixed to `New folder (6)/template`, `New folder (6)/selfie`, and `New folder (6)/tests`.

## Official references used
- PaddleOCR general OCR pipeline and optional modules:
  - https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html
  - https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/doc_preprocessor.html
- PaddleOCR KIE / ID-card extraction direction:
  - https://paddlepaddle.github.io/PaddleOCR/v2.9/en/ppstructure/blog/how_to_do_kie.html
  - https://paddlepaddle.github.io/PaddleOCR/v2.9/en/ppstructure/model_train/train_kie.html
- OpenCV planar-object localization with homography:
  - https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
  - https://docs.opencv.org/3.4/dd/dd4/tutorial_detection_of_planar_objects.html

## Files
- `id_card_pipeline_researched.py`: main pipeline
- `run_batch_from_folders_newfolder6_researched.py`: batch runner
- `init_project_and_test_newfolder6_researched.py`: one-click setup + batch run
- `requirements_researched.txt`: pinned dependencies
