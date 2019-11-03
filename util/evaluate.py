import torch
import args


def evaluate_classification(model, dev_data):
    losses = []
    device = args.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:
            input_ids, input_mask, segment_ids, score = batch.input_ids, batch.input_mask, batch.segment_ids, batch.score
            loss = model(input_ids.to(device), segment_ids.to(
                device), input_mask.to(device), score.to(device))
            loss = loss.mean()
            losses.append(loss.item())
        total = sum(losses)
        print("eval_loss: " + str(total / len(losses)))

        return total / len(losses)


def evaluate_answer(model, dev_data):
    losses = []
    device = args.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device), segment_ids.to(device), input_mask.to(
                device), start_positions.to(device), end_positions.to(device))
            loss = loss.mean()
            losses.append(loss.item())
        total = sum(losses)
        print("eval_loss: " + str(total / len(losses)))

        return total / len(losses)
