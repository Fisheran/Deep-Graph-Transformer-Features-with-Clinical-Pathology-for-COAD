from __future__ import annotations
import os, torch
from timm.models import create_model

# Tool function for verifying model loading
def check_model_loaded(model, state, device, fp16=True, missing=None, unexpected=None):
    """
    Verify whether the model has loaded successfully
    Directly print the validation report and return pass/fail status
    
    Parameters:
        model: The model after loading weights
        state: Weight dictionary
        device: Running device
        fp16: Whether to use half-precision
        missing: List of missing parameters
        unexpected: list of extra parameters
    
    Returns: bool - True indicates successful loading, False indicates issues
    """
    print("\n" + "="*72)
    print("Model loading verification report".center(72))
    print("="*72)

    is_valid = True
    
    matched = len(state) - len(unexpected)
    
    # 1. Weighted Matching Degree
    print(f"\n【1. Weighted Matching Degree】")
    print(f"  ✓ Successfully loaded: {matched} parameters")

    if len(missing) == 0:
        print(f"  ✓ Missing: 0")
    else:
        print(f"  ✗ Missing: {len(missing)}")
        is_valid = False
        for key in list(missing)[:3]:
            print(f"      - {key}")
        if len(missing) > 3:
            print(f"      ... {len(missing)-3} more")
    
    # Inspection unexpected
    if len(unexpected) == 0:
        print(f"  ✓ superfluous: 0")
    else:
        # Check for common ignorable items
        ignorable_prefixes = ['optimizer', 'scheduler', 'epoch', 'step', 
                             'best', 'loss', 'config', 'timestamp', 'iteration']
        ignorable = [k for k in unexpected if any(k.startswith(p) for p in ignorable_prefixes)]
        non_ignorable = [k for k in unexpected if k not in ignorable]
        
        if len(non_ignorable) == 0:
            print(f"  ✓ superfluous: {len(unexpected)} (all training info, ignorable)")
        elif len(non_ignorable) < len(unexpected) * 0.1:  # less than 10%
            print(f"  ⚠ superfluous: {len(unexpected)} (mostly ignorable)")
            if non_ignorable:
                print(f"    Caution:")
                for key in list(non_ignorable)[:3]:
                    print(f"      - {key}")
        else:
            print(f"  ⚠ superfluous: {len(unexpected)} (Possible mismatch in weighting files!)")
            print(f"    The first five:")
            for key in list(unexpected)[:5]:
                print(f"      - {key}")
            if len(non_ignorable) > len(unexpected) * 0.5:
                # If more than half are non-ignorable, mark as invalid
                is_valid = False
    
    # 2. Weighted Health Check
    nan_count = sum(1 for p in model.parameters() if torch.isnan(p).any())
    inf_count = sum(1 for p in model.parameters() if torch.isinf(p).any())
    zero_count = sum(1 for p in model.parameters() if torch.all(p == 0))

    print(f"\n【2. Weighted Health Check】")
    print(f"  {'✓' if nan_count==0 else '✗'} NaN: {nan_count}")
    print(f"  {'✓' if inf_count==0 else '✗'} Inf: {inf_count}")
    print(f"  {'✓' if zero_count==0 else '⚠'} zero: {zero_count}")
    
    if nan_count > 0 or inf_count > 0:
        is_valid = False
    
    # 3. Forward Pass Test
    print(f"\n【3. Forward Pass Test】")
    try:
        # Retrieve the device on which the model is actually located
        model_device = next(model.parameters()).device

        # Create test input, ensuring it's on the same device as the model
        test_input = torch.randn(1, 3, 384, 384).to(model_device)
        if fp16:
            test_input = test_input.half()
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  ✓ Test passed")
        
        main_output = output[0]
        print(f"  ✓ Output type: tuple (containing {len(output)} elements)")
        print(f"  ✓ Main output shape: {list(main_output.shape)}")

        # Move output to CPU for further inspection
        main_output_cpu = main_output.detach().cpu()
        print(f"  ✓ Main output range: [{main_output_cpu.min().item():.4f}, {main_output_cpu.max().item():.4f}]")

        # Check all outputs for abnormalities
        has_nan = False
        has_inf = False
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                out_cpu = out.detach().cpu()
                if torch.isnan(out_cpu).any():
                    has_nan = True
                    print(f"  ✗ Output[{i}] contains NaN")
                if torch.isinf(out_cpu).any():
                    has_inf = True
                    print(f"  ✗ Output[{i}] contains Inf")
                    
        if has_nan or has_inf:
            is_valid = False
            
    except Exception as e:
        print(f"  ✗ Test failed: {str(e)[:80]}")
        is_valid = False

    # Summary
    print(f"\n{'='*70}")
    if is_valid:
        print("✓ Model loaded successfully!".center(70))
    else:
        print("✗ Model failed to load, please check the issues above".center(70))
    print("="*72 + "\n")
    
    return is_valid

# Tool function for verifying model loading
def load_musk_model(ckpt_path=None, device=None, fp16=True):
    """
    Load the MUSK model from a local checkpoint.
    Parameters:
        ckpt_path: Path to the local 'model.safetensors' checkpoint file.
        device: Device to load the model onto ('cuda' or 'cpu'). If None, auto-detects.
        fp16: Whether to convert the model to half-precision (float16).

    Returns:
        Tuple[torch.nn.Module, str]: The loaded MUSK model and the device it's on.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to import musk.modeling to register model name; otherwise build explicitly
    model = None
    try:
        # registers timm models
        model = create_model("musk_large_patch16_384")
    except Exception as ex:
        # Direct constructor fallback (requires musk installed)
        try:
            from MUSK.musk.modeling import musk_large_patch16_384
            model = musk_large_patch16_384()
        except Exception as ex2:
            raise RuntimeError("Cannot import MUSK model. Please install the MUSK repo (pip install -e <path>) or ensure MUSK.modeling is importable.") from ex2

    # Load local safetensors (recommended)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError("MUSK_CKPT not found. Please set MUSK_CKPT to a local 'model.safetensors'.")
    from safetensors.torch import load_file
    state = load_file(ckpt_path)
    # strip common prefixes
    state = {k.replace("model.","").replace("module.",""): v for k,v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    # Verify whether the model has loaded successfully
    is_valid = check_model_loaded(model=model, state=state, device=device, fp16=fp16, missing=missing, unexpected=unexpected)
    if not is_valid:
        print("⚠ Warning: An issue has been detected, but the model has been loaded. It is recommended to check it before use.")
    
    model = model.to(device).eval()
    if fp16:
        model = model.half()
    
    return model, device        