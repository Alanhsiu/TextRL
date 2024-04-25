# Text-Guided-Voice-Conversion

Paper: https://arxiv.org/abs/2309.14324

## Convert speech to neural audio codec. 

See **waveform_to_unit.py** .

## Training the autoregressive model.

See **trainer_encodec_vc_ar.py** .

## Training the non-autoregressive model.

See **trainer_encodec_vc_nar.py** .

## Inference.

See **trainer_encodec_vc_inference.py** .

## Model Checkpoint (Hugging Face)

### (1) Autoregressive Model

#### (a) Text-pretrained
```
lca0503/speech-chatgpt-base-ar-v2-epoch4-wotrans
```

#### (b) Text-to-speech-pretrained
```
lca0503/speech-chatgpt-base-ar-v2-epoch4-wotrans-tts
```

#### (c) Train from scratch
```
lca0503/speech-chatgpt-base-ar-v2-epoch4-wotrans-scratch
```

### (2) Non-autoregressive Model

#### (a) Text-pretrained
```
lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans
```

#### (b) Text-to-speech-pretrained
```
lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans-tts
```

#### (c) Train from scratch
```
lca0503/speech-chatgpt-base-nar-v2-epoch4-wotrans-scratch
```

