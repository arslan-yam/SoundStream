import torch


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

    audios = [item["audio"] for item in dataset_items]
    max_len = max(a.shape[-1] for a in audios)
    lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    padded = []
    for a in audios:
        if a.shape[-1] < max_len:
            pad = max_len - a.shape[-1]
            a = torch.nn.functional.pad(a, (0, pad))
        padded.append(a)
    audio = torch.stack(padded, dim=0)
    return {"audio": audio, "audio_lengths": lengths}
