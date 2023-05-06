from model_utils import get_dh_from_top_k_important_messages
import torch

def test_get_dh_from_top_k_important_messages():
    # Create sample h_message and importance_scores tensors
    h_message = torch.rand(2, 5, 6, 4)  # shape: [a, b, c, d]
    importance_scores = torch.rand(2, 5, 6, 1)  # shape: [a, b, c, 1]

    k = 3
    dh = get_dh_from_top_k_important_messages(h_message, importance_scores, k)

    # Check if the output shape is correct
    assert dh.shape == (2, 5, 4), f"Unexpected output shape: {dh.shape}"

    # Check if the values in dh are within the range of h_message
    assert (dh.min() >= h_message.min()) and (dh.max() <= h_message.max()), "Output values out of range"

test_get_dh_from_top_k_important_messages()