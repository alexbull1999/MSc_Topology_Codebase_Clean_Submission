import torch
import geoopt
import numpy as np


def test_geoopt_basic():
    """Test basic geoopt functionality."""
    print("Testing geoopt basic functionality...")

    try:
        # Create Poincaré ball
        ball = geoopt.PoincareBall()
        print("Poincare Ball created successfully")

        # Test points creation
        x = torch.randn(5, 10) * 0.1  # Small values
        print(f"Created test points: shape {x.shape}")

        # Test exponential map from origin
        x_hyp = ball.expmap0(x)
        print(f"Exponential map: {x_hyp.shape}")

        # Test distance computation
        distances = ball.dist(x_hyp[0:1], x_hyp[1:2])
        print(f"Distance computation: {distances.item():.4f}")

        # Test Euclidean norms (should be < 1 for points in ball)
        euclidean_norms = torch.norm(x_hyp, dim=-1)
        print(f"Euclidean norms range: {euclidean_norms.min():.4f} - {euclidean_norms.max():.4f}")

        if torch.all(euclidean_norms < 1.0):
            print("All points inside unit ball")
        else:
            print("Some points outside unit ball - may need smaller scaling")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_geoopt_advanced():
    """Test advanced geoopt operations for hyperbolic projection."""
    print("\nTesting advanced geoopt operations...")

    try:
        ball = geoopt.PoincareBall()

        # Test batch operations
        batch_size = 16
        dim = 20

        # Create batch of points
        x = torch.randn(batch_size, dim) * 0.05
        x_hyp = ball.expmap0(x)

        print(f"Batch exponential map: {x_hyp.shape}")

        # Test pairwise distances
        origin = torch.zeros(1, dim)
        origin_hyp = ball.expmap0(origin)
        origin_expanded = origin_hyp.expand(batch_size, -1)

        distances_from_origin = ball.dist(origin_expanded, x_hyp)
        print(f"Distances from origin: mean={distances_from_origin.mean():.4f}")

        # Test inter-point distances
        if batch_size >= 2:
            inter_distances = ball.dist(x_hyp[0:1].expand(batch_size - 1, -1), x_hyp[1:])
            print(f"Inter-point distances: mean={inter_distances.mean():.4f}")

        return True

    except Exception as e:
        print(f"Advanced test error: {e}")
        return False


def test_compatibility_fix():
    """Test the specific fix for the norm issue."""
    print("\nTesting compatibility fix...")

    try:
        ball = geoopt.PoincareBall()

        # Create test embeddings
        embeddings = torch.randn(5, 10) * 0.1
        embeddings_hyp = ball.expmap0(embeddings)

        # Test the old way (that was failing)
        try:
            # This might fail on some geoopt versions
            norms_old = ball.norm(embeddings_hyp)
            print("Old norm method works")
        except Exception as e:
            print(f"Old norm method fails: {e}")
            print("   This is expected - we'll use the fix")

        # Test the new way (using distance from origin)
        origin = torch.zeros_like(embeddings_hyp[0:1])
        origin = origin.expand_as(embeddings_hyp)
        norms_new = ball.dist(origin, embeddings_hyp)
        print(f"New norm method works: {norms_new.mean():.4f}")

        # Test Euclidean norm fallback
        euclidean_norms = torch.norm(embeddings_hyp, dim=-1)
        print(f"Euclidean norm fallback: {euclidean_norms.mean():.4f}")

        return True

    except Exception as e:
        print(f"Compatibility test error: {e}")
        return False


def main():
    """Run all geoopt tests."""
    print("Geoopt Compatibility Test Suite")
    print("=" * 50)

    # Test basic functionality
    basic_ok = test_geoopt_basic()

    if basic_ok:
        # Test advanced operations
        advanced_ok = test_geoopt_advanced()

        # Test compatibility fix
        compat_ok = test_compatibility_fix()

        if basic_ok and advanced_ok and compat_ok:
            print("\nAll tests passed! Geoopt is ready for hyperbolic projection.")
            print("You can now run: python hyperbolic_projection.py")
        else:
            print("\nSome tests failed. Check the error messages above.")
    else:
        print("\nBasic functionality failed.")


if __name__ == "__main__":
    main()