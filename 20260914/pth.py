from pathlib import Path

import torch


MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATHS = (
	MODEL_DIR / "translation_model_big.pth",
	MODEL_DIR / "translation_model.pth",
)


def print_state_dict(name, state_dict):
	print(f"\n{name} parameters:")
	total_params = 0

	for parameter_name, parameter in state_dict.items():
		if not torch.is_tensor(parameter):
			print(f"  {parameter_name}: {type(parameter).__name__}")
			continue

		parameter_count = parameter.numel()
		total_params += parameter_count
		print(
			f"  {parameter_name}: shape={tuple(parameter.shape)}, "
			f"dtype={parameter.dtype}, params={parameter_count:,}"
		)

	print(f"Total parameters: {total_params:,}")
	return total_params


def inspect_checkpoint(model_path):
	print("=" * 80)
	print(f"Loading: {model_path}")

	if not model_path.exists():
		print("File not found")
		return None

	checkpoint = torch.load(model_path, map_location="cpu")
	print(f"Checkpoint type: {type(checkpoint).__name__}")

	if not isinstance(checkpoint, dict):
		print("This checkpoint is not a dictionary.")
		return checkpoint

	print(f"Checkpoint keys: {list(checkpoint.keys())}")
	for key in ("embedding_size", "hidden_size", "best_epoch", "best_loss"):
		if key in checkpoint:
			print(f"{key}: {checkpoint[key]}")

	encoder_state = checkpoint.get("encoder")
	decoder_state = checkpoint.get("decoder")

	if isinstance(encoder_state, dict):
		print_state_dict("Encoder", encoder_state)
		if "embedding.weight" in encoder_state:
			print(
				"Encoder vocabulary size: ",
				encoder_state["embedding.weight"].shape[0]
			)
	else:
		print("Encoder state_dict not found")

	if isinstance(decoder_state, dict):
		print_state_dict("Decoder", decoder_state)
		if "embedding.weight" in decoder_state:
			print(
				"Decoder vocabulary size: ",
				decoder_state["embedding.weight"].shape[0]
			)
	else:
		print("Decoder state_dict not found")

	if "optimizer" in checkpoint and isinstance(checkpoint["optimizer"], dict):
		optimizer_state = checkpoint["optimizer"]
		print(
			"Optimizer state: "
			f"{len(optimizer_state.get('state', {}))} parameter states, "
			f"param_groups={len(optimizer_state.get('param_groups', []))}"
		)

	return checkpoint


def main():
	checkpoints = []
	for model_path in MODEL_PATHS:
		checkpoint = inspect_checkpoint(model_path)
		if checkpoint is not None:
			checkpoints.append((model_path.name, checkpoint))

	print("\n" + "=" * 80)
	print("Embedding size comparison:")
	for model_name, checkpoint in checkpoints:
		encoder = checkpoint.get("encoder", {})
		decoder = checkpoint.get("decoder", {})
		encoder_embedding = encoder.get("embedding.weight")
		decoder_embedding = decoder.get("embedding.weight")
		print(
			f"{model_name}: "
			f"encoder={tuple(encoder_embedding.shape) if torch.is_tensor(encoder_embedding) else 'N/A'}, "
			f"decoder={tuple(decoder_embedding.shape) if torch.is_tensor(decoder_embedding) else 'N/A'}"
		)


if __name__ == "__main__":
	main()
