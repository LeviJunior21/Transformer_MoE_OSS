import torch, time, wandb
from .calc_loss import calc_loss_batch, evaluate_model, generate_and_print_sample

def train_model(
        model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer,
        wandb_run, save_freq_wdb, file_name, save_wdb, state_dict, project, name, start_time
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    epochs_complete = state_dict.get("epoch", 0)
    batchs_complete = state_dict.get("batch", 0)
    accumulated_time = state_dict.get("train_time", 0)

    for epoch in range(epochs_complete, num_epochs):
        model.train()

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            if epoch == epochs_complete and batch_idx < batchs_complete: continue

            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                wandb_run.log({
                    "train_loss": train_loss, "val_loss": val_loss, "tokens_seen": tokens_seen, "epoch": epoch + 1, "global_step": global_step})

                print(f"Ep {epoch+1} (Step {global_step:06d}): \nTrain loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if save_wdb and global_step >0 and global_step % save_freq_wdb == 0:
                elapsed_time = time.time() - start_time + accumulated_time
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "batch": batch_idx, "train_time": elapsed_time},  file_name)
                artifact = wandb.Artifact(name, type="model")
                artifact.add_file(file_name)
                wandb_run.log_artifact(artifact)


        print(f"\nEXEMPLO DE GERAÇÃO: {generate_and_print_sample(model, tokenizer, device, start_context)}")
        elapsed_time = time.time() - start_time + accumulated_time
        if save_wdb:
          torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "batch": batch_idx, "train_time": elapsed_time},  file_name)
          artifact = wandb.Artifact(file_name.split(".")[0] + "_final", type="model")
          artifact.add_file(file_name)
          wandb_run.log_artifact(artifact)

    return train_losses, val_losses, track_tokens_seen, elapsed_time