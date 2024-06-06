import math

def calculate_pv(revenue_0, growth_rate, discount_rate, max_growth_rate=1.5, min_growth_rate=-0.7, probability_of_loss=0.1, time_horizon=12):
    """
    Calculates the present value (PV) of a lifetime revenue stream.

    Args:
        revenue_0 (float): Initial revenue.
        growth_rate (float): Predicted annual growth rate.
        discount_rate (float): Discount rate used for PV calculation.
        max_growth_rate (float, optional): Maximum allowed growth rate. Defaults to 0.5.
        min_growth_rate (float, optional): Minimum allowed growth rate. Defaults to -0.2.
        probability_of_loss (float, optional): Probability of losing the revenue stream for negative growth rates. Defaults to 0.1.
        time_horizon (int, optional): Number of years to consider for the revenue stream. Defaults to 20.

    Returns:
        float: Present value of the lifetime revenue stream.
    """
    pv = 0
    revenue = revenue_0

    for t in range(time_horizon):
        # Constrain growth rate to the specified range
        growth_rate = max(min(growth_rate, max_growth_rate), min_growth_rate)

        # Calculate revenue for the current year
        if growth_rate >= 0:
            revenue *= (1 + growth_rate)
        else:
            # Incorporate the probability of losing the revenue stream
            revenue *= (1 + growth_rate) * (1 - probability_of_loss)

        # Calculate present value for the current year
        discount_factor = 1 / (1 + discount_rate) ** (t + 1)
        pv += revenue * discount_factor

    return pv

print(calculate_pv(10, -1000, 0.05))