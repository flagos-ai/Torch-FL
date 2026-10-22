#!/usr/bin/env python3
"""Live test of MUSA vendor routing in ProcessGroupFlagOS.

This test verifies that:
1. GEMS_VENDOR=musa is correctly recognized
2. MUSA profile uses identity view
3. Routing logic attempts FlagCX then MCCL
4. Appropriate error messages when backends unavailable

Does NOT require actual multi-GPU communication, just backend initialization.
"""

import os
import sys

# Force MUSA vendor
os.environ["GEMS_VENDOR"] = "musa"


def test_musa_profile():
    """Verify MUSA vendor profile configuration."""
    from torch_fl.comm.process_group import _VENDOR_PROFILES, _get_profile

    print("=" * 60)
    print("Test 1: MUSA Vendor Profile")
    print("=" * 60)

    # Check MUSA profile exists
    assert "musa" in _VENDOR_PROFILES, "MUSA profile missing"

    prof = _get_profile("musa")
    print("✓ MUSA profile found")
    print(f"  flagcx_dev: {prof.flagcx_dev}")
    print(f"  view: {prof.view}")
    print(f"  native: {prof.native}")

    assert prof.flagcx_dev == "musa", f"Wrong flagcx_dev: {prof.flagcx_dev}"
    assert prof.view == "_flagos_identity_view", f"Wrong view: {prof.view}"
    assert prof.native == "_try_build_mccl", f"Wrong native: {prof.native}"

    print("✓ MUSA profile correct: musa + identity view + MCCL fallback")


def test_identity_view_binding():
    """Verify _flagos_identity_view C++ binding is accessible."""
    import torch
    import torch_fl

    print("\n" + "=" * 60)
    print("Test 2: Identity View Binding")
    print("=" * 60)

    torch_fl._C._init()

    # Test identity view
    t = torch.randn(3, 4, device="flagos:0")
    viewed = torch_fl._C._flagos_identity_view(t)

    assert viewed is t, "Identity view must return same object"
    assert viewed.data_ptr() == t.data_ptr(), "Data pointer must match"

    print("✓ _flagos_identity_view works correctly")
    print(f"  Same object: {viewed is t}")
    print(f"  Same data_ptr: {hex(viewed.data_ptr())} == {hex(t.data_ptr())}")


def test_backend_routing():
    """Test ProcessGroupFlagOS initialization routing logic."""
    import torch_fl
    from torch_fl.comm.process_group import ProcessGroupFlagOS

    print("\n" + "=" * 60)
    print("Test 3: Backend Routing Logic")
    print("=" * 60)

    torch_fl._C._init()

    # Attempt to create ProcessGroupFlagOS
    # This will fail because neither FlagCX nor MCCL backend is available,
    # but we can verify the error message is correct for MUSA
    try:
        # Mock store for testing (won't actually communicate)
        from torch.distributed import TCPStore

        store = TCPStore("127.0.0.1", 0, 1, True)

        ProcessGroupFlagOS(store, 0, 1)
        print("✗ Unexpected success - should have failed with no backend")
        sys.exit(1)
    except RuntimeError as e:
        error_msg = str(e)
        print("✓ Got expected RuntimeError")
        print(f"  Message: {error_msg}")

        # Verify error message mentions MUSA and MCCL
        assert "GEMS_VENDOR='musa'" in error_msg, (
            f"Error should mention MUSA vendor: {error_msg}"
        )
        assert "_try_build_mccl" in error_msg, (
            f"Error should mention MCCL fallback: {error_msg}"
        )

        print("✓ Error message correctly identifies MUSA and MCCL requirement")


def main():
    print("MUSA Vendor Routing Live Test")
    print("=" * 60)
    print(f"GEMS_VENDOR: {os.environ.get('GEMS_VENDOR')}")
    print("=" * 60)

    try:
        test_musa_profile()
        test_identity_view_binding()
        test_backend_routing()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print("  ✓ MUSA vendor profile correctly configured")
        print("  ✓ Identity view binding accessible and working")
        print("  ✓ Routing logic attempts FlagCX → MCCL for MUSA")
        print("\nTo enable actual distributed communication:")
        print("  - Install FlagCX with MUSA adaptor, OR")
        print("  - Install torch_musa with ProcessGroupMCCL")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
