"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

MODEL_ID = 'FireRedTeam/FireRedLID'
# Hugging Face commit of the model weights
MODEL_REVISION = '1bb4d285c8456429385d9c0810300df4297bc11b'


def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="FireRed Spoken LID",
        description=(
            "Spoken language identification over fixed-length windows of "
            "an audio track, using FireRedLID, the language identification "
            "model of FireRedASR2S. Consecutive windows with the same "
            "language can be merged into one TimeFrame."),
        app_license="Apache 2.0",
        identifier="firered-spoken-lid",
        url="https://github.com/clamsproject/app-firered-spoken-lid",
        analyzer_versions={MODEL_ID: MODEL_REVISION},
        analyzer_license="Apache 2.0",
        # measured on an RTX A6000 with 10 s windows: about 4 GB reserved
        # by torch at batch size 1 or 8, plus the CUDA context
        est_gpu_mem_min=4500,
        est_gpu_mem_typ=4500,
    )
    metadata.add_input_oneof(DocumentTypes.AudioDocument,
                             DocumentTypes.VideoDocument)
    out_tf = metadata.add_output(AnnotationTypes.TimeFrame,
                                 timeUnit='milliseconds', document='*',
                                 labelset='*', label='*', scores='*')
    out_tf.add_description(
        'Each TimeFrame covers consecutive windows that FireRedLID labels '
        'with the language `label`, or `other`. Language labels are '
        'canonicalized to BCP 47 tags, and `labelMap` can reduce them; see '
        'README.md in the app repository for details. `labelset` lists all '
        'labels the view can have. `scores` lists, in time order, the '
        'FireRedLID score of `label` in each of those '
        'windows, one float in [0, 1] per window. Window i starts at '
        '`start` + i * `windowMs`; the last window ends at `end` and can be '
        'shorter. The scores are not calibrated.')

    metadata.add_parameter(
        name='windowMs', type='integer', default=10000,
        description='Window length in milliseconds.')
    metadata.add_parameter(
        name='minLastWindowMs', type='integer', default=0,
        description='Shortest final window to keep, in milliseconds. 0 '
                    'keeps any final window.')
    metadata.add_parameter(
        name='tfMerge', type='boolean', default=True,
        description='Join consecutive windows with the same label into one '
                    'TimeFrame. When false, each window becomes its own '
                    'TimeFrame, with one score.')
    metadata.add_parameter(
        name='labelMap', type='string', default=[], multivalued=True,
        description='Labels to keep, as "IN:OUT" (rename IN to OUT) or "IN" '
                    '(keep IN as it is). Other labels become "-". Empty '
                    'keeps every label of the model. See README.md in the '
                    'app repository for how the label is chosen.')
    metadata.add_parameter(
        name='minLabelScore', type='number', default=0.5,
        description='Minimum score for a label from labelMap to be kept; a '
                    'window whose best mapped label scores less becomes '
                    '"-". Only applies with labelMap.')
    metadata.add_parameter(
        name='batchSize', type='integer', default=1,
        description='Number of windows passed to the model at once.')
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
