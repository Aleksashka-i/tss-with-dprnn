## Target Speech Separation Demonstration

This tiny demonstration of target speech separation is a part of my bachelor thesis. Below are examples of the mixture, target, and reference audio.

| <center>Mixture</center> | <center>Target</center> | <center>Reference</center> |
| :---: | :---: | :---: |
|<audio src="audio/mix.wav" controls preload></audio>|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/reference.wav" controls preload>|
|<img src="spectrograms/mix.png"/>|<img src="spectrograms/target.png"/>|<img src="spectrograms/reference.png"/>|

## Estimated audio
The estimated audio samples were obtained using the DPRNN-Spe model. Below are examples of different fusion types, including Addition, Attention, Concatenation, FiLM, and Multiplication.

| | <center>Target</center> | <center>Concatenation</center> | <center>Attention</center> |  <center>Multiplication</center> | <center>FiLM</center> | <center>Addition</center> |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Spectrogram |<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_add.png"/>|<img src="spectrograms/estimated_spe_att.png"/>|<img src="spectrograms/estimated_spe_mul.png"/>|<img src="spectrograms/estimated_spe_film.png"/>|<img src="spectrograms/estimated_spe_add.png"/>|
| Audio |<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_cat.wav" controls preload></audio>|<audio src="audio/estimated_spe_att.wav" controls preload></audio>|<audio src="audio/estimated_spe_mul.wav" controls preload></audio>|<audio src="audio/estimated_spe_film.wav" controls preload></audio>|<audio src="audio/estimated_spe_add.wav" controls preload></audio>|
| <center>SI-SDR</center> | -- | 18.50 | 17.40 | 17.13 | 9.97 | 9.34 |

## Close-up examples
Below are close-up examples of different fusion types:

| <center>Target</center> | <center>Concatenation</center> | 
| :---: | :---: |
|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_cat.wav" controls preload></audio>|
|<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_cat.png"/>|

| <center>Target</center> | <center>Attention</center> |
| :---: | :---: |
|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_att.wav" controls preload></audio>|
|<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_att.png"/>|

| <center>Target</center> | <center>Multiplicatio</center> |
| :---: | :---: |
|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_mul.wav" controls preload></audio>|
|<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_mul.png"/>|

| <center>Target</center> | <center>FiLM</center> |
| :---: | :---: |
|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_film.wav" controls preload></audio>|
|<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_film.png"/>|

| <center>Target</center> | <center>Addition</center> |
| :---: | :---: |
|<audio src="audio/target.wav" controls preload></audio>|<audio src="audio/estimated_spe_add.wav" controls preload></audio>|
|<img src="spectrograms/target.png"/>|<img src="spectrograms/estimated_spe_add.png"/>|
