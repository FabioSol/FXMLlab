from dataclasses import dataclass
from datetime import datetime
from typing import List
import pandas as pd
from config import BACKTESTER_CONFIG as BC

@dataclass
class Order:
    type: str  # 0 long; 1 short
    open_price: float
    open_time: datetime
    tp: float
    sl: float


@dataclass
class Journal:
    balance: float
    open_orders: List[Order]
    closed_orders: List[Order]


def back_test(df: pd.DataFrame, col: str, bullish_cols: list, bearish_cols: list, sl_long: float, tp_long: float, sl_short: float, tp_short: float, size: int):
    journal = Journal(BC.initial_balance, [], [])
    historic = pd.DataFrame(index=df.index,columns=["values"])
    for idx, row in df.iterrows():
        price = row[col]
        bullish = row[bullish_cols].fillna(0).values
        bearish = row[bearish_cols].fillna(0).values
        effective_price = price * size
        min_balance = 0
        short_orders = []

        # close orders
        for order in journal.open_orders[::-1]:
            if order.type == "sell":
                if (order.sl <= price) or (order.tp >= price):
                    journal.closed_orders += [order]
                    journal.balance -= effective_price * (1 + BC.COMMISSION)
                    journal.open_orders.remove(order)
                else:
                    min_balance += effective_price * BC.MARGIN
                    short_orders += [order]
            else:
                if (order.sl >= price) or (order.tp <= price):
                    journal.closed_orders += [order]
                    journal.balance += effective_price * (1 - BC.COMMISSION)
                    journal.open_orders.remove(order)

        while min_balance > journal.balance:
            if len(short_orders) > 0:
                order = short_orders.pop(0)
                journal.open_orders.remove(order)
                journal.balance -= effective_price * (1 + BC.COMMISSION)
                min_balance -= effective_price * BC.MARGIN
            else:
                order = journal.open_orders.pop(0)
                journal.balance += effective_price * (1 - BC.COMMISSION)
            journal.closed_orders += [order]

        # check if there are funds
        if len(journal.open_orders) < 50:
            if journal.balance > max(effective_price, min_balance):
                mean = sum(bullish) / len(bullish) - sum(bearish) / len(bearish)
                if mean >= 0:
                    journal.open_orders += [
                        Order("buy", price, idx, price * (1 + tp_long / (100)), price * (1 - sl_long / (100)))]
                    journal.balance -= effective_price
                elif mean <= 0:
                    journal.open_orders += [
                        Order("sell", price, idx, price * (1 - tp_short / (100)), price * (1 + sl_short / (100)))]
                    journal.balance += effective_price

        value = journal.balance-len([order for order in journal.open_orders if order.type =="sell"])*effective_price+\
                len([order for order in journal.open_orders if order.type =="buy"])*effective_price
        historic.loc[idx] = {"values": value}

    last_effective_price = df[col].iloc[-1] * size
    for order in journal.open_orders[::-1]:
        if order.type == "sell":
            journal.closed_orders += [order]
            journal.balance -= last_effective_price * (1 + BC.COMMISSION)
            journal.open_orders.remove(order)
        else:
            journal.closed_orders += [order]
            journal.balance += last_effective_price * (1 - BC.COMMISSION)
            journal.open_orders.remove(order)

    return historic, Journal, journal.balance / BC.initial_balance - 1



