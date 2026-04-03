    def apply_market_friction(
        self,
        target_price: float,
        order_size_usd: float,
        side: str = "buy",
        asset: str = "BTCUSDT"
    ) -> Dict[str, float]:
        """
        Apply all market friction models to determine final execution price.
        
        Args:
            target_price: Intended execution price
            order_size_usd: Order size in USDT
            side: "buy" or "sell"
            asset: Asset symbol
            
        Returns:
            Dictionary with breakdown:
            {
                'target_price': float,
                'slippage': float,
                'latency_impact': float,
                'liquidity_impact': float,
                'execution_price': float,
                'fee': float,
                'total_cost': float  # Including fees
            }
        """
        if not self.enable_market_friction:
            # No friction - return target price with basic fee
            basic_fee = order_size_usd * 0.001  # 0.1%
            return {
                'target_price': target_price,
                'slippage': 0.0,
                'latency_impact': 0.0,
                'liquidity_impact': 0.0,
                'execution_price': target_price,
                'fee': basic_fee,
                'total_cost': order_size_usd + (basic_fee if side == "buy" else -basic_fee)
            }
        
        # Estimate market conditions (simplified - could fetch from environment state)
        market_conditions = MarketConditions(
            volatility=0.01,  # Default 1% volatility
            spread_bps=5.0,
            volume_24h=1e9
        )
        
        # Apply each friction model sequentially
        price_after_slippage = self.slippage_model.apply_slippage(
            target_price, order_size_usd, market_conditions, side
        )
        slippage = price_after_slippage - target_price
        
        price_after_latency = self.latency_model.apply_latency(
            price_after_slippage, market_conditions
        )
        latency_impact = price_after_latency - price_after_slippage
        
        final_price = self.liquidity_model.apply_impact(
            price_after_latency, order_size_usd, market_conditions, side
        )
        liquidity_impact = final_price - price_after_latency
        
        # Calculate fee
        order_value = order_size_usd
        fee = self.fee_model.calculate_fee(order_value, is_maker=False)
        
        # Total cost accounting
        if side == "buy":
            total_cost = (order_size_usd / target_price) * final_price + fee
        else:
            total_cost = (order_size_usd / target_price) * final_price - fee
        
        return {
            'target_price': float(target_price),
            'slippage': float(slippage),
            'latency_impact': float(latency_impact),
            'liquidity_impact': float(liquidity_impact),
            'execution_price': float(final_price),
            'fee': float(fee),
            'total_cost': float(total_cost)
        }
