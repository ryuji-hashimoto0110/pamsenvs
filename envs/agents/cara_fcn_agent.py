from pams.logs.base import ExecutionLog
from ..markets import TotalTimeAwareMarket
from ..markets import YesterdayAwareMarket
import math
import numpy as np
from numpy import ndarray
from pams.agents import Agent
from pams.logs import Logger
from pams.market import Market
from pams.order import Cancel
from pams.order import LIMIT_ORDER
from pams.order import MARKET_ORDER
from pams.order import Cancel
from pams.order import Order
from pams.order import OrderKind
from pams.simulator import Simulator
from pams.utils import JsonRandom
from scipy import optimize
from typing import Any, Optional
from typing import TypeVar
import random
import warnings

AgentID = TypeVar("AgentID")
MarketID = TypeVar("MarketID")

class CARAFCNAgent(Agent):
    """FCN Agent w/ CARA utility (CARAFCNAgent) class

    CARAFCNAgent's order decision mechanism is based on Chiarella et al., 2008.

    References:
        - Chiarella, C., Iori, G., & Perello, J. (2009). The impact of heterogeneous trading rules
        on the limit order book and order flows,
        Journal of Economic Dynamics and Control, 33 (3), 525-537. https://doi.org/10.1016/j.jedc.2008.08.001

    note: The original paper consider market order, but CARAFCNAgent here is not allowed to submit market order
        because it unstabilize the simulation due to the specification of PAMS.
    """
    def __init__(
        self,
        agent_id: AgentID,
        prng: random.Random,
        simulator: Simulator,
        name: str,
        logger: Optional[Logger]
    ) -> None:
        super().__init__(agent_id, prng, simulator, name, logger)
        self.unexecuted_orders: list[Order] = []

    def is_finite(self, x: float) -> bool:
        """determine if it is a valid value.

        Args:
            x (float): value.

        Returns:
            bool: whether or not it is a valid (not NaN, finite) value.
        """
        return not math.isnan(x) and not math.isinf(x)

    def setup(
        self,
        settings: dict[str, Any],
        accessible_markets_ids: list[MarketID],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """agent setup. Usually be called from simulator / runner automatically.

        CARAFCNAgent allows Pareto distribution for initial cash amount of agents. See _convert_exp2pareto method.

        Args:
            settings (dict[str, Any]): agent configuration. Thie must include the parameters:
                - fundamentalWeight: weight given to the fundamentalist component.
                - chartWeight: weight given to the chartist component.
                - noiseWeight: weight given to the chartist component.
                - noiseScale: the scale of noise component.
                - timeWindowSize: time horizon.
                - isCARA: wheter order decision is based on CARA utility. If False,
                    basic FCN order decision is applied.
                - riskAversionTerm: reference level of risk aversion.
                    The precise relative risk aversion coefficient is calculated
                    by using fundamental/chart weights.
                and can include
                - orderMargin: must be set if isCARA is set false.
                - chartFollowRate: probability that the agent is chart-follower.
                - yesterdayAware: whether the agent know yesterday market prices.
                    If yesterdayAware is set true, accessible market must be YesterdayAwareMarket.
            accessible_market_ids (list[MarketID]): _description_

        If feedbackAsymmetry and noiseAsymmetry are both 0, aFCNAgent is equivalent to FCNAgent.
        """
        self.settings: dict[str, Any] = settings
        if not hasattr(self, "cashAmount"):
            super().setup(
                settings=settings, accessible_markets_ids=accessible_markets_ids
            )
            self._convert_exp2pareto(settings)
        if 2 <= len(accessible_markets_ids):
            warnings.warn(
                "order decision for multiple assets has not implemented yet."
            )
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.w_f: float = json_random.random(json_value=settings["fundamentalWeight"])
        if "averageCashAmount" in settings:
            average_cash_amount: float = settings["averageCashAmount"]
            cash_amount: float = self.cash_amount
            self.w_f *= cash_amount / average_cash_amount
        self.w_c: float = json_random.random(json_value=settings["chartWeight"])
        self.w_n: float = json_random.random(json_value=settings["noiseWeight"])
        self.noise_scale: float = json_random.random(json_value=settings["noiseScale"])
        self.time_window_size = int(
            json_random.random(json_value=settings["timeWindowSize"])
        )
        self.is_cara: bool = settings["isCARA"]
        if not self.is_cara:
            if "orderMargin" not in settings:
                raise ValueError(
                    "orderMargin must be set when isCARA is false."
                )
            base_order_margin: float = json_random.random(
                json_value=settings["orderMargin"]
            )
            if "averageTimeWindowSize" in settings:
                average_tau: float = settings["averageTimeWindowSize"]
                self.order_margin: float = base_order_margin * (
                    self.time_window_size / average_tau
                )
            else:
                self.order_margin: float = base_order_margin
        if "riskAversionTerm" in settings:
            self.risk_aversion_term: float = json_random.random(
                json_value=settings["riskAversionTerm"]
            )
        else:
            self.risk_aversion_term: float = 0.1
        if "meanReversionTime" in settings:
            self.mean_reversion_time: int = int(
                json_random.random(json_value=settings["meanReversionTime"])
            )
        else:
            self.mean_reversion_time: int = self.time_window_size
        if "yesterdayAware" in settings:
            self.is_yesterday_aware: bool = settings["yesterdayAware"]
        else:
            self.is_yesterday_aware: bool = False
        if "chartFollowRate" in settings:
            p: float = settings["chartFollowRate"]
            if p < 0 or 1 < p:
                raise ValueError(
                    f"chartFollowRate must be in between [0,1]."
                )
            self.is_chart_following: bool = self.prng.choices(
                [True, False], weights=[p, 1-p]
            )[0]
        else:
            self.is_chart_following: bool = True

    def _convert_exp2pareto(self, settings: dict[str, Any]) -> None:
        """convert Exponential distribution to Pareto distribution.

        Default pams does not allow Pareto distribution. This method convert some specified variables with
        Exponential distribution to Pareto distribution. The following is an example to set config.

        {
            "assetVolume": {"expon": [50]},
		    "cashAmount": {"expon": [100]},
            "paretoVariables: {
                "cashAmount": {"alpha": 1.0, "beta": 1.0},
                "assetVolume": {"alpha": ..., "beta": ...}
            }
        }
        """
        if "paretoVariables" in settings:
            pareto_variables: dict[str, dict[str, float]] = settings["paretoVariables"]
        else:
            return
        for pareto_variable, param_dic in pareto_variables.items():
            if "expon" not in settings[pareto_variable]:
                raise ValueError(
                    f"inappropriate distribution type to convert to pareto: {settings[pareto_variable]}"
                )
            lam: float = 1 / settings[pareto_variable]["expon"][0]
            alpha: float = param_dic["alpha"]
            beta: float = param_dic["beta"]
            if pareto_variable == "cashAmount":
                cash_amount: float = lam * self.cash_amount
                cash_amount = alpha * np.exp(cash_amount / beta)
                self.set_cash_amount(cash_amount)
            elif pareto_variable == "assetVolume":
                for market_id in self.asset_volumes.keys():
                    asset_volume: float = lam * self.asset_volumes[market_id]
                    asset_volume: int = int(alpha * np.exp(asset_volume / beta))
                    self.set_asset_volume(market_id, asset_volume)
            else:
                raise NotImplementedError

    def submit_orders(
        self, markets: list[Market]
    ) -> list[Order | Cancel]:
        """submit orders based on FCN-based calculation.
        """
        orders: list[Order | Cancel] = sum(
            [
                self.submit_orders_by_market(market=market) for market in markets
            ], []
        )
        return orders

    def submit_orders_by_market(self, market: Market) -> list[Order | Cancel]:
        """submit orders by market (internal usage).

        CARAFCNAgent submit orders by following procedure.
            1. cancel orders remaining unexecuted in the market.
            2. calculate temporal FCN weights, time window size and risk aversion term.
            3. calculate expected future price by FCN rule.
            4. calculate expected volatility and temporal risk aversion.
            5. create new order using the demand function induced from CARA utility.

        Args:
            market (Market): market to order.

        Returns:
            orders (list[Order | Cancel]): order list
        """
        orders: list[Order | Cancel] = []
        if not self.is_market_accessible(market_id=market.market_id):
            return orders
        orders.extend(self._cancel_orders())
        time: int = market.get_time()
        time_window_size: int = self._calc_temporal_time_window_size(
            time, self.w_f, self.w_c, self.w_n, market
        )
        weights: list[float] = self._calc_temporal_weights(market, time_window_size)
        fundamental_weight: float = weights[0]
        chart_weight: float = weights[1]
        noise_weight: float = weights[2]
        assert 0 <= fundamental_weight
        assert 0 <= chart_weight
        assert 0 <= noise_weight
        risk_aversion_term: float = self._calc_temporal_risk_aversion_term(
            fundamental_weight, chart_weight, noise_weight
        )
        assert 0 <= time_window_size
        assert 0 < risk_aversion_term
        expected_future_price: float = self._calc_expected_future_price(
            market, weights, time_window_size
        )
        assert self.is_finite(expected_future_price)
        expected_volatility: float = self._calc_expected_volatility(
            market, time_window_size
        )
        assert self.is_finite(expected_volatility)
        orders.extend(
            self._create_order(
                market, expected_future_price, expected_volatility,
                time_window_size, risk_aversion_term
            )
        )
        return orders

    def _calc_temporal_weights(
        self,
        market: Market,
        time_window_size: int
    ) -> list[float]:
        """calculate temporal FCN weights.

        Args:
            market (Market): market to order.
            time_window_size (int): time window size

        Returns:
            weights(list[float]): weights list. [fundamental weight, chartist weight, noise weight]
        """
        weights: list[float] = [self.w_f, self.w_c, self.w_n]
        return weights

    def _calc_temporal_time_window_size(
        self,
        time: int,
        fundamental_weight: float,
        chart_weight: float,
        noise_weight: float,
        market: Market,
        **kwargs
    ) -> int:
        """calculate temporal time window size.

        Assume that agent time horizon depends on its charactetistics. In detail, agent who emphasize
        fundamentalist strategy typically have longer time horizon. On the other hand, agent mostly rely on
        chartist strategy, referring short term price fluctuation, tend to have shorter time horizon.

        Args:
            time_window_size (int): time horizon.
            time (int): market time.
            fundamental_weight (float): reference level of the agent's fundamental weight.
            chart_weight (float): reference level of the agent's chart weight.

        Returns:
            temporal_time_window_size (int): calculated the agent's temporal time horizon.
        """
        time_window_size: int = int(
            self.time_window_size * (
                (1 + fundamental_weight) / (1 + chart_weight + noise_weight)
            )
        )
        if self.is_yesterday_aware:
            return time_window_size
        else:
            return min(time, time_window_size)

    def _calc_temporal_risk_aversion_term(
        self,
        fundamental_weight: float,
        chart_weight: float,
        noise_weight: float
    ) -> float:
        """calculate temporal relative risk aversion term in CARA utility.

        Args:
            fundamental_weight (float): temporal fundamental weight.
            chart_weight (float): temporal chart weight.

        Returns:
            risk_aversion_term (float): calculated the agent's temporal risk aversion term.
        """
        risk_aversion_term: float = self.risk_aversion_term * (
            (1 + fundamental_weight) / (1 + chart_weight + noise_weight)
        )
        return risk_aversion_term

    def _calc_expected_future_price(
        self,
        market: Market,
        weights: list[float],
        time_window_size: int
    ) -> float:
        """calculate expected future price by FCN rule.

        ..seealso:
            - :func: `pams.agents.FCNAgent.submit_orders_by_market'
        """
        fundamental_weight: float = weights[0]
        chart_weight: float = weights[1]
        noise_weight: float = weights[2]
        time: int = market.get_time()
        market_price: float = market.get_market_price()
        fundamental_price: float = market.get_fundamental_price()
        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return: float = fundamental_scale * math.log(
            fundamental_price / market_price
        )
        assert self.is_finite(fundamental_log_return)
        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_log_return: float = chart_scale * math.log(
            market_price / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_log_return)
        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)
        expected_log_return: float = (
            1.0 / (fundamental_weight + chart_weight + noise_weight)
        ) * (
            fundamental_weight * fundamental_log_return
            + chart_weight * chart_log_return * (1 if self.is_chart_following else -1)
            + noise_weight * noise_log_return
        )
        assert self.is_finite(expected_log_return)
        expected_future_price: float = market_price * math.exp(
            expected_log_return * time_window_size
        )
        return expected_future_price

    def _calc_expected_volatility(
        self,
        market: Market,
        time_window_size: int
    ) -> float:
        """calculate expected volatility.

        CARAFCNAgent estimate volatility as the variabce of past log returns.
        If order execution is not allowed in current session, the market price never changes.
        In such a case, replace expected volatility to sufficientlly small value: 1e-10.

        Args:
            market (Market): market to order.
            time_window_size (int): time horizon.

        Returns:
            float: expectef volatility
        """
        time: int = market.get_time()
        market_prices: list[float] = market.get_market_prices(
            range(time-time_window_size,time+1)
        )
        log_returns: ndarray = np.log(market_prices[1:]) - np.log(market_prices[:len(market_prices)-1])
        avg_log_return: float = np.sum(log_returns) / (time_window_size + 1e-10)
        expected_volatility: float = np.sum((log_returns - avg_log_return)**2) / (time_window_size + 1e-10)
        assert self.is_finite(expected_volatility)
        expected_volatility = max(1e-10, expected_volatility)
        return expected_volatility

    def _create_order(
        self,
        market: Market,
        expected_future_price: float,
        expected_volatility: float,
        time_window_size: int,
        risk_aversion_term: float
    ) -> list[Order | Cancel]:
        """create new orders.
        """
        if time_window_size < 1:
            time_window_size = self.time_window_size
        if self.is_cara:
            orders: list[Order | Cancel] = self._create_order_cara(
                market, expected_future_price, expected_volatility,
                time_window_size, risk_aversion_term
            )
        else:
            orders: list[Order | Cancel] = self._create_order_wo_cara(
                market, expected_future_price, time_window_size
            )
        for order in orders:
            if isinstance(order, Order):
                self.unexecuted_orders.append(order)
        return orders
    
    def _create_order_cara(
        self,
        market: Market,
        expected_future_price: float,
        expected_volatility: float,
        time_window_size: int,
        risk_aversion_term: float
    ) -> list[Order | Cancel]:
        """create new orders w/ CARA utility.

        This method create new order according to the demand of the agent indiced from CARA utility
        in following procedure.
            1. estimate numerically the price level at which the agent is satisfied with the composition
                of his or her current portfolio.
            2. set the maximum selling price p_M at which demand(p_M) being 0 to ensure that
                short selling is not allowed.
            3. set the minimum buying price p_m at which p_m (demand(p_m) - current_stock_position) is equal to
                current cash position to impose budget constraint.
            4. Having determined the interval [p_m, p_M] in which the agent is willing to trade,
                randomly draw a price from the interval and decide order type and volume according to the demand.

        Args:
            market (Market): market to order.
            expected_future_price (float): expected future price.
            expected_volatility (float): expected volatility.
            risk_aversion_term (float): temporal risk aversion term.

        Returns:
            orders (list[Order | Cancel]): created orders to submit
        """
        asset_volume: int = self.get_asset_volume(market.market_id)
        cash_amount: float = self.get_cash_amount()
        lower_bound: float = 1e-10
        satisfaction_price: float = optimize.brentq(
            self._calc_additional_demand,
            a=lower_bound, b=expected_future_price,
            args=(expected_future_price, risk_aversion_term,
                expected_volatility, asset_volume)
        )
        max_sell_price: float = expected_future_price
        try:
            min_buy_price: float = optimize.brentq(
                self._calc_remaining_cash,
                a=lower_bound, b=satisfaction_price,
                args=(expected_future_price, risk_aversion_term,
                    expected_volatility, asset_volume, max(0, cash_amount))
            )
        except Exception as e:
            min_buy_price: float = satisfaction_price
        assert min_buy_price <= satisfaction_price
        assert satisfaction_price <= max_sell_price
        price: Optional[float] = self.prng.uniform(min_buy_price, max_sell_price)
        order_kind: OrderKind = LIMIT_ORDER
        if price < satisfaction_price:
            is_buy: bool = True
            best_sell_price: float = market.get_best_sell_price()
            if best_sell_price is None:
                best_sell_price = market.get_market_price()
            if best_sell_price < price:
                best_sell_price: float = np.clip(best_sell_price, min_buy_price, max_sell_price)
                price = best_sell_price
                demand: float = self._calc_demand(
                    best_sell_price , expected_future_price, risk_aversion_term, expected_volatility
                )
            else:
                demand: float = self._calc_demand(
                    price, expected_future_price, risk_aversion_term, expected_volatility
                )
            order_volume: int = int(demand - asset_volume)
        else:
            is_buy: bool = False
            best_buy_price: float = market.get_best_buy_price()
            if best_buy_price is None:
                best_buy_price = market.get_market_price()
            if price < best_buy_price:
                best_buy_price: float = np.clip(best_buy_price, min_buy_price, max_sell_price)
                price = best_buy_price
                demand: float = self._calc_demand(
                    best_buy_price, expected_future_price, risk_aversion_term, expected_volatility
                )
            else:
                demand: float = self._calc_demand(
                    price, expected_future_price, risk_aversion_term, expected_volatility
                )
            order_volume: int = int(asset_volume - demand)
        orders: list[Order | Cancel] = []
        if not order_volume == 0:
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=market.market_id,
                    is_buy=is_buy,
                    kind=order_kind,
                    volume=order_volume,
                    price=price,
                    ttl=time_window_size
                )
            )
        return orders

    def _calc_demand(
        self,
        price: float,
        expected_future_price: float,
        risk_aversion_term: float,
        expected_volatility: float
    ) -> float:
        """demand function. D(price|expected_future_price, risk_aversion_term, expected_volatility)

        Args:
            price (float): price level at which the demand is calculated.
            expected_future_price (float): expected future price. constant variable.
            risk_aversion_term (float): temporal risk aversion term. constant variable.
            expected_volatility (float): expected_volatility. constant variable.

        Returns:
            demand (float): calculate demand.
        """
        demand: float = (
            np.log(expected_future_price / price)
        ) / (
            risk_aversion_term * expected_volatility * price
        )
        return demand

    def _calc_additional_demand(
        self,
        price: float,
        expected_future_price: float,
        risk_aversion_term: float,
        expected_volatility: float,
        asset_volume: int
    ) -> float:
        """calculate additional demand.

        Additional demand means the amount of stock that the agent is willing to buy (sell, if negative)
        at the given price level: D(price) - current_stock_position.

        Args:
            price (float): price level at which the additional demand is calculated.
            expected_future_price (float): expected future price. constant variable.
            risk_aversion_term (float): temporal risk aversion term. constant variable.
            expected_volatility (float): expected_volatility. constant variable.
            asset_volume (int): currently holding asset position.

        Returns:
            additional_demand (float): calculated additional demand.
        """
        demand: float = self._calc_demand(
            price, expected_future_price, risk_aversion_term, expected_volatility
        )
        additional_demand: float = demand - asset_volume
        return additional_demand

    def _calc_remaining_cash(
        self,
        price: float,
        expected_future_price: float,
        risk_aversion_term: float,
        expected_volatility: float,
        asset_volume: int,
        cash_amount: float
    ) -> float:
        """calculate remaining cash.

        remaining cash means the cash volume remained if the agent buy (additional_demand(price)) units of
        stocks at the given price level:
            current_cash_position - price * (D(price) - current_stock_position)

        Args:
            price (float): price level at which the remaining cash is calculated.
            expected_future_price (float): expected future price. constant variable.
            risk_aversion_term (float): temporal risk aversion term. constant variable.
            expected_volatility (float): expected_volatility. constant variable.
            asset_volume (int): currently holding asset position.
            cash_amount (float): currently holding cash position.

        Returns:
            remaining_cash (float): calculated remaining cash.
        """
        buying_price: float = price * (
            self._calc_demand(
                price, expected_future_price, risk_aversion_term, expected_volatility
            ) - asset_volume
        )
        remaining_cash: float = cash_amount - buying_price
        return remaining_cash

    def _cancel_orders(self) -> list[Cancel]:
        """cancel orders remaining unexecuted in the market.
        """
        cancels: list[Cancel] = []
        for order in self.unexecuted_orders:
            if not order.volume == 0:
                cancels.append(Cancel(order))
        self.unexecuted_orders = []
        return cancels
    
    def _create_order_wo_cara(
        self,
        market: Market,
        expected_future_price: float,
        time_window_size: int
    ) -> list[Order | Cancel]:
        """create new orders w/o CARA utility.

        This method create new order according to only the expected future price.

        Args:
            market (Market): market to order.
            expected_future_price (float): expected future price.

        Returns:
            orders (list[Order | Cancel]): created orders to submit
        """
        orders: list[Order | Cancel] = []
        market_price: float = market.get_market_price()
        order_volume: int = 1
        order_kind: OrderKind = LIMIT_ORDER
        if market_price < expected_future_price:
            is_buy: bool = True
            order_price: float = expected_future_price * (1 - self.order_margin)
        else:
            is_buy: bool = False
            order_price: float = expected_future_price * (1 + self.order_margin)
        orders.append(
            Order(
                agent_id=self.agent_id,
                market_id=market.market_id,
                is_buy=is_buy,
                kind=order_kind,
                volume=order_volume,
                price=order_price,
                ttl=time_window_size
            )
        )
        return orders
