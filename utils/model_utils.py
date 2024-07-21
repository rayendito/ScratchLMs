
# stole this from chatgpt lol
def show_parameter_counts(model):
    total_params = 0
    # print(f"{'Layer':<40} {'Parameters':>15}")
    print("="*55)
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_count = parameter.numel()
        # print(f"{name:<40} {param_count:>15}")
        total_params += param_count
        
    print(f"Total Trainable Parameters: {total_params}")
    print("="*55)
    