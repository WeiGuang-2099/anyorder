"""Buyers competing for limited stock.

The naive buyer reads the stock and writes it back one lower in separate
tool calls, so two buyers can both see the last unit. The atomic buyer makes
one reserve call that checks and decrements together. The two scenarios
differ only in the buyer, which makes the atomic one the control: a failure
reported there is a bug in the tester.
"""

from __future__ import annotations

import functools

from anyorder import scheduler


class Inventory:
    """Stock of one item and the orders placed against it."""

    def __init__(self, stock: int) -> None:
        self.initial_stock = stock
        self.stock = stock
        self.orders: list[str] = []

    def read_stock(self) -> int:
        return self.stock

    def write_stock(self, value: int) -> None:
        self.stock = value

    def place_order(self, buyer: str) -> None:
        self.orders.append(buyer)

    def reserve(self, buyer: str) -> bool:
        """Takes one unit for the buyer if any is left, in one call."""
        if self.stock <= 0:
            return False
        self.stock -= 1
        self.orders.append(buyer)
        return True


async def naive_buyer(ctx: scheduler.AgentContext) -> None:
    stock = await ctx.call("read_stock")
    if stock > 0:
        await ctx.call("write_stock", value=stock - 1)
        await ctx.call("place_order", buyer=ctx.name)


async def atomic_buyer(ctx: scheduler.AgentContext) -> None:
    await ctx.call("reserve", buyer=ctx.name)


def no_oversell(inventory: Inventory) -> str | None:
    """Fails when more orders were placed than there was stock."""
    if len(inventory.orders) <= inventory.initial_stock:
        return None
    return (
        f"{len(inventory.orders)} orders placed against a stock of "
        f"{inventory.initial_stock}: {', '.join(inventory.orders)}")


def naive(stock: int = 1, buyers: int = 2) -> scheduler.Scenario:
    """Buyers that check and decrement in separate calls."""
    return _scenario("oversell-naive", naive_buyer, stock, buyers)


def atomic(stock: int = 1, buyers: int = 2) -> scheduler.Scenario:
    """The control: buyers that check and decrement in one call."""
    return _scenario("oversell-atomic", atomic_buyer, stock, buyers)


def _scenario(
    name: str, buyer: scheduler.Agent, stock: int, buyers: int
) -> scheduler.Scenario:
    return scheduler.Scenario(
        name=name,
        make_world=functools.partial(Inventory, stock),
        agents={f"buyer-{i}": buyer for i in range(1, buyers + 1)},
        invariants=(
            scheduler.Invariant("no-oversell", no_oversell, every_step=True),
        ),
    )
