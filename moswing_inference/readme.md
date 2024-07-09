# MosWing: Mosquito Wingbeat Sound Event Detection

## Install Dependencies

Please use python 3.10.12. Use only either conda or pip.

### conda

```bash
conda env create -f env.yml
```

### pip

```bash
pip install -r requirements.txt
```

## Inference

```bash
python inference.py $MODEL_DIR $SOUND_DIR --output_dir $OUTPUT_DIR
```

The code above will load model from `$MODEL_DIR` and sound from `$SOUND_DIR` then output the prediction to the `$OUTPUT_DIR`.

`$MODEL_DIR` is a path to directory that contain `settings.json` and a `.h5` file. The `settings.json` must contain `model_name`, `habitat_setting`, and `thresholds`. `$SOUND_DIR` must be a directory that contain sound file (.wav) only. `$OUTPUT_DIR` will be create during runtime.


### Examples

```bash
python inference.py model_graveyard/m181_aecx test_sound/SED-Test --output_dir out/m181
```

```bash
python inference.py model_graveyard/m198_ancx test_sound/SED-Test --output_dir out/m198
```
