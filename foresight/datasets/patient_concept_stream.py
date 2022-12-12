from __future__ import absolute_import, division, print_function

import pickle
import logging
import datasets
import numpy as np


_CITATION = """\
@misc{kraljevic2020multidomain,
      title={Multi-domain Clinical Natural Language Processing with MedCAT: the Medical Concept Annotation Toolkit}, 
      author={Zeljko Kraljevic and Thomas Searle and Anthony Shek and Lukasz Roguski and Kawsar Noor and Daniel Bean and Aurelie Mascio and Leilei Zhu and Amos A Folarin and Angus Roberts and Rebecca Bendayan and Mark P Richardson and Robert Stewart and Anoop D Shah and Wai Keong Wong and Zina Ibrahim and James T Teo and Richard JB Dobson},
      year={2020},
      eprint={2010.01165},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Takes as input a pickled dict of pt2stream. The format should be:
    {'patient_id': (concept_cui, concept_count_for_patient, timestamp_of_first_occurrence_for_patient), ...}
"""

class PatientConceptStreamConfig(datasets.BuilderConfig):
    """ BuilderConfig for PatientConceptStream.

        Args:
            **kwargs: keyword arguments forwarded to super.
    """

    def __init__(self, **kwargs):
        super(PatientConceptStreamConfig, self).__init__(**kwargs)


class PatientConceptStream(datasets.GeneratorBasedBuilder):
    """PatientConceptStream: as input takes the patient to stream of concepts.

    TODO: Move the preparations scripts out of notebooks
    """

    BUILDER_CONFIGS = [
        PatientConceptStreamConfig(
            name="pickle",
            version=datasets.Version("1.0.0", ""),
            description="Pickled output from Temporal dataset preparation scripts",
        ),
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "patient_id": datasets.Value("string"),
                    "stream": [
                        {
                            'token': datasets.Value('string'),
                            'cnt': datasets.Value('int32'),
                            'time': datasets.Value('int64'),
                            'token_type': datasets.Value('string'),
                            #'context_representation': datasets.Sequence(datasets.Value('float'), length=128),
                        }
                    ],
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepaths': self.config.data_files['train'],
                },
            ),
        ]

    def _generate_examples(self, filepaths):
        """Returns Pts one by one"""
        for filepath in filepaths:
            logging.info("generating examples from = %s", filepath)
            with open(filepath, 'rb') as f:
                pt2stream = pickle.load(f)
                for pt, stream in pt2stream.items():
                    out_stream = []
                    # If time == None there is no temporal info, and for this dataset it is required
                    stream = [data for data in stream if data[2] is not None]
                    # Sort the stream by time - ascending
                    stream.sort(key=lambda data: data[2])
                    for data in stream:
                        out_stream.append({
                                           'token': data[0], # Call it token from now on as it does not have to be only CUIs
                                           'cnt': data[1],
                                           'time': int(data[2]), # We convert this into int for speed
                                           'token_type': data[3], # Call it token from now on as it does not have to be only CUIs
                                           #'context_representation': np.random.rand(128), #data[4],
                                           })
                    pt = str(pt)
                    if out_stream: # Skip streams that have zero annotations
                        yield pt, {'patient_id': str(pt),
                                   'stream': out_stream}
