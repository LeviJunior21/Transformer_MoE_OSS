import torch
from ...src.utils.tokenizer import text_to_token_ids, token_ids_to_text
from ...src.utils.generate import generate_text


def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(model, data_loader, device, num_batches):
    total_loss = 0.0
    if len(data_loader) == 0: return float('nan')
    num_batches = min(num_batches, len(data_loader))
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            input_batch, target_batch = next(data_iter)
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        except StopIteration: break
    return total_loss / num_batches if num_batches > 0 else float('nan')


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Amostra Gerada: '{decoded_text.replace(os.linesep, ' ')}'")
    model.train()