# Detecting and Classifying Whale Calls with Wavelet Scattering and Spectral Entropy
Resources for paper submitted the Journal of the Acoustical Society of America (JASA).

``scattering/`` contains the custom wavelet scattering implementation.

``detect/`` contains the detectors.

``scripts/`` contains all scripts and utilities required to produce the results from this paper.

``results/`` contains the raw result output from our evaluation.

``fig/`` contains the figures used in the manuscript.

``scripts/compute_tf.py`` is required to pre-compute the STFT and WS decompositions for detections. Detectors are evaluated in ``scripts/evaluate_detectors.py``, after which classification is evaluated on the proposed detector in ``scripts/evaluate_classifier.py``. ``scripts/parameters.py`` contains all the settings for detectors and classifier.

``casey2017subset.tar.gz`` contains the audio data and repaired annotations (along with a RavenPro workspace file) used for evaluation in this paper. You will need to reimport the audio files in the RavenPro workspace.
