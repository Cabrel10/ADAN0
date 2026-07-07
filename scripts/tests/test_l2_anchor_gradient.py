"""Numerical unit test for the V15 L2 action anchor gradient.

Verifies the mathematical claim: anchor_loss = lambda * (mu**2).mean() produces
a gradient on mu equal to (2*lambda/N)*mu, i.e. a linear restoring force toward 0.
This is the property the reward-level V14 anchor could NOT provide (its gradient
was diluted by GAE/VF). No env, no training loop — pure autograd on the formula.
"""
import sys
import torch

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    print("=== V15 L2 ANCHOR GRADIENT UNIT TESTS ===")
    lam = 0.05

    # T1: gradient of lambda*(mu**2).mean() wrt mu is (2*lambda/N)*mu
    N = 8
    mu = torch.tensor([6.23, 5.0, 1.0, 0.5, 0.0, -0.5, -3.0, -6.23], requires_grad=True)
    loss = lam * (mu ** 2).mean()
    loss.backward()
    expected = (2.0 * lam / N) * mu.detach()
    check("T1 gradient == (2*lambda/N)*mu",
          torch.allclose(mu.grad, expected, atol=1e-7),
          f"max|err|={float((mu.grad-expected).abs().max()):.2e}")

    # T2: restoring force always points toward 0 (sign(grad) == sign(mu))
    nz = mu.detach() != 0
    same_sign = (torch.sign(mu.grad[nz]) == torch.sign(mu.detach()[nz])).all()
    check("T2 gradient points toward 0 (opposes mu after -lr*grad step)", bool(same_sign))

    # T3: magnitude grows with |mu| -> the further mu fled, the stronger the pull
    g = mu.grad.abs()
    order_ok = (g[0] > g[2]) and (g[2] > g[4])  # |mu|=6.23 > 1.0 > 0.0
    check("T3 |grad| monotone in |mu|", bool(order_ok),
          f"g(6.23)={float(g[0]):.4f} g(1.0)={float(g[2]):.4f} g(0)={float(g[4]):.4f}")

    # T4: a single SGD step reduces |mu| for the runaway value (6.23)
    lr = 0.01
    mu2 = torch.tensor([6.23], requires_grad=True)
    l2 = lam * (mu2 ** 2).mean()
    l2.backward()
    with torch.no_grad():
        new = mu2 - lr * mu2.grad
    check("T4 one SGD step shrinks mu=6.23", float(new.abs()) < 6.23,
          f"6.23 -> {float(new):.5f}")

    # T5: lambda=0 => zero gradient contribution (safe A-B control)
    mu3 = torch.tensor([6.23, -6.23], requires_grad=True)
    l3 = 0.0 * (mu3 ** 2).mean()
    l3.backward()
    check("T5 lambda=0 => zero grad", torch.allclose(mu3.grad, torch.zeros_like(mu3)),
          f"grad={mu3.grad.tolist()}")

    # T6: anchor magnitude vs a typical PPO policy-loss magnitude sanity.
    # At mu=6.23, per-element anchor loss = lambda*mu^2 = 0.05*38.8 = 1.94 -> LARGE,
    # so once mu has fled it is strongly penalised (unlike the reward -0.02 cap).
    val = lam * (6.23 ** 2)
    check("T6 anchor loss at mu=6.23 is O(1) not O(0.01)", val > 1.0,
          f"anchor_loss(mu=6.23)={val:.3f}")

    print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("RESULT: ALL L2 ANCHOR GRADIENT TESTS PASSED")


if __name__ == "__main__":
    main()
