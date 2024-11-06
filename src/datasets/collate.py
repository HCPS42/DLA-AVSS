import torch
from torch.utils.data import default_collate


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    def get_type_keys(type: str):
        """
        Every item in the instance is one of 4 types:

        - Path to the file on the disk. Has suffix `_path`
        - Audio waveform. Has suffix `_wav`
        - Spectrogram. Has suffix `_spec`
        - Mouth landmarks. Has suffix `_npz`

        Possible prefixes are `mix`, `speaker1`, `speaker2` for waveforms and spectrograms.

        This function filters all keys of certain type from the dataset items
        since we want to collate objects depending on their type.

        Args:
            type (str): type of fields.
        Returns:
            keys (list[str]): list of keys of the type.
        """
        return list(
            filter(lambda key: key.endswith(f"_{type}"), dataset_items[0].keys())
        )

    path_keys = get_type_keys("path")
    wav_keys = get_type_keys("wav")
    spec_keys = get_type_keys("spec")
    # npz_keys = get_type_keys("npz")

    result_batch = {}

    # collate paths
    result_batch.update(
        default_collate(
            list(map(lambda item: {key: item[key] for key in path_keys}, dataset_items))
        )
    )

    # collate waveforms
    result_batch.update(
        {
            key: torch.nn.utils.rnn.pad_sequence(
                [item[key].squeeze(0) for item in dataset_items], batch_first=True
            )
            for key in wav_keys
        }
    )

    # collate waveforms lengths
    result_batch.update(
        {
            f"{key}_length": default_collate(
                [
                    item[key].shape[-1]
                    for item in dataset_items
                    # shape of waveform is (1 x L)
                ]
            )
            for key in wav_keys
        }
    )

    # collate spectrograms
    result_batch.update(
        {
            # collated spectrograms have shape (bs x L x n_mels)
            key: torch.nn.utils.rnn.pad_sequence(
                [item[key].squeeze(0).T for item in dataset_items], batch_first=True
            )
            for key in spec_keys
        }
    )

    # collate spectrograms lengths
    result_batch.update(
        {
            f"{key}_length": default_collate(
                [
                    item[key].shape[-1]
                    for item in dataset_items
                    # shape of spectrogram is (n_mels x L)
                ]
            )
            for key in spec_keys
        }
    )

    # we don't put npz in the batch in current version
    # (audio only)

    return result_batch
