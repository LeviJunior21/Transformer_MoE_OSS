import torch, wandb, time
from ...src.utils.graphs import plot_graph
from ...src.utils.tokenizer import tokenizer
from .load_weights import fetch_weights_and_bias, load_weights_and_bias

def train_model_aux(model, optimizer, config, train_loader, val_loader, device):
    run = wandb.init(project=config["project"], name=config["name"], id=config["run_id"], resume="allow")
    res_fetch = fetch_weights_and_bias(user=config["user"], project=config["project"], name=config["name"], version=config["version"], file_name=config["file_name"])
    
    state_dict = {}
    if res_fetch:
        loaded, checkpoint = load_weights_and_bias(file_name=config["file_name"])

        if loaded:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            state_dict["epoch"] = checkpoint.get("epoch", 0)
            state_dict["batch"] = checkpoint.get("batch", 0)
            state_dict["train_time"] = checkpoint.get("train_time", 0.0)
            print("Pesos carregados com sucesso!")

    print(f"\nEPOCHS/BATCHS RECUPERADOS: {state_dict}")
    num_epochs = config["max_epochs"]
    start_time = time.time()
    train_losses, val_loss, tokens_seen, total_train_time = train_model_aux(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=config["start_context"], tokenizer=tokenizer,
        wandb_run=run, save_freq_wdb=config["save_freq_wdb"], file_name=config["file_name"],
        save_wdb=config["save_wdb"], state_dict=state_dict,
        project=config["project"], name=config["name"], start_time=start_time
    )

    print("\nGRÁFICO DE PERDA DURANTE O TREINO:")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_graph(epochs_tensor, tokens_seen, train_losses, val_loss)
    return tokens_seen[-1], total_train_time