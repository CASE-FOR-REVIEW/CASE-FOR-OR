import numpy as np
import pandas as pd
from scipy import optimize
from typing import Tuple, List

class DDFeasibilitySolver:
    """
    Simple DD  feasibility solver.
    
    Checks:  
    f_theta_1(x_pre) > r 
    AND 
    f_theta_1(x_post) <= r 

    """
    
    def solve_batch(self, 
                   risks_pre_df, 
                   risks_post_df,
                   r: float,
                   alpha
                   ) -> pd.DataFrame:
        """
        Solve robust feasibility for multiple options.
        
        Args:

            risks_pre_df: DataFrame where each row is an option, each column is a 
                         pre-treatment risk scenario f_theta_i(x_pre)
            risks_post_df: DataFrame where each row is an option, each column is a
                          post-treatment risk scenario f_theta_i(x_post)
            r: Threshold value
            
        Returns:
            Boolean Series indicating feasibility for each option
        """
        
        # VECTORIZED: Check constraints (equivalent to your implementation)
        pre_feasible =  (risks_pre_df > r)
        post_feasible = (risks_post_df <= r)
        
        # Robust feasibility: both constraints satisfied
        feasibility = np.logical_and(pre_feasible.iloc[:, 0], post_feasible.iloc[:, 0])
                
        results_df = pd.DataFrame({"is_feasible": feasibility})
        print(f"Completed! Found {results_df['is_feasible'].sum()} feasible options for DD.")
        
        return results_df
    
class SDDFeasibilitySolver:
    """
    Simple SDD  feasibility solver.
    
    Checks:  
    Percentage(Indicator f_theta_i(x_pre) > r) >= 1-alpha 
    AND 
    Percentage(Indicator f_theta_i(x_post) <= r) >= 1-alpha 

    """
    
    def solve_batch(self, 
                   risks_pre_df, 
                   risks_post_df,
                   r: float,
                   alpha) -> pd.DataFrame:
        """
        Solve robust feasibility for multiple options.
        
        Args:

            risks_pre_df: DataFrame where each row is an option, each column is a 
                         pre-treatment risk scenario f_theta_i(x_pre)
            risks_post_df: DataFrame where each row is an option, each column is a
                          post-treatment risk scenario f_theta_i(x_post)
            rho: Parameter rho
            r: Threshold value
            alpha: Confidence level alpha  
            
        Returns:
            Boolean Series indicating feasibility for each option
        """
        
        # VECTORIZED: Check constraints (equivalent to your implementation)
        pre_feasible = (risks_pre_df > r).mean(axis=1) >= 1 - alpha 
        post_feasible = (risks_post_df <= r).mean(axis=1) >= 1 - alpha
        
        # Robust feasibility: both constraints satisfied
        feasibility = pre_feasible & post_feasible
        
        results_df = pd.DataFrame({"is_feasible": feasibility})
        print(f"Completed! Found {results_df['is_feasible'].sum()} feasible options for SDD.")
        
        return results_df
    
class RDFeasibilitySolver:
    """
    Simple robust design feasibility solver.
    
    Checks: f_theta_i(x_pre) > r AND f_theta_i(x_post) <= r, for all i = 1,...,N
    """
    
    def solve_batch(self, 
                   risks_pre_df, 
                   risks_post_df,
                   r: float,
                   alpha) -> pd.DataFrame:
        """
        Solve robust feasibility for multiple options.
        
        Args:

            risks_pre_df: DataFrame where each row is an option, each column is a 
                         pre-treatment risk scenario f_theta_i(x_pre)
            risks_post_df: DataFrame where each row is an option, each column is a
                          post-treatment risk scenario f_theta_i(x_post)
            r: Threshold value

            
        Returns:
            Boolean Series indicating feasibility for each option
        """
        
        # VECTORIZED: Check constraints (equivalent to your implementation)
        pre_feasible = (risks_pre_df > r).mean(axis=1) >= 1  # ALL models > r
        post_feasible = (risks_post_df <= r).mean(axis=1) >= 1  # ALL models <= r
        
        # Robust feasibility: both constraints satisfied
        feasibility = pre_feasible & post_feasible

        results_df = pd.DataFrame({"is_feasible": feasibility})
        print(f"Completed! Found {results_df['is_feasible'].sum()} feasible options for RD.")
        
        return results_df
    
class DRDFeasibilitySolver:
    """
    Simple solver for the chance-constrained feasibility problem:
    
    sup_{lambda>=0} { lambda rho + 1/N Σᵢ phi*(lambda Indicator{f_theta_i(x_pre) <=r})} <=alpha
    sup_{lambda>=0} { lambda rho + 1/N Σᵢ phi*(lambda Indicator{f_theta_i(x_post) > r})} <=alpha
    
    Clean implementation: vectorized operations + scipy optimization + simple loops.
    
    ## Example usage
    # Generate sample data
    np.random.seed(42)
    n_options = 50      # Number of investment options/strategies
    n_scenarios = 100   # Number of risk scenarios
    
    print("=== Simple Chance-Constrained Feasibility Solver ===")
    print(f"Problem size: {n_options} options x {n_scenarios} scenarios")
    
    # Create DataFrames: rows = options, columns = scenarios
    risks_pre_df = pd.DataFrame(
        np.random.gamma(2, 0.5, (n_options, n_scenarios)),
        index=[f'Strategy_{i}' for i in range(n_options)],
        columns=[f'Scenario_{j}' for j in range(n_scenarios)]
    )
    
    risks_post_df = pd.DataFrame(
        np.random.gamma(1.5, 0.6, (n_options, n_scenarios)), 
        index=[f'Strategy_{i}' for i in range(n_options)],
        columns=[f'Scenario_{j}' for j in range(n_scenarios)]
    )
    
    # Problem parameters
    rho = 0.01
    r = 0.075
    alpha = 0.05
    
    print(f"Parameters: rho={rho}, r={r}, alpha={alpha}")
    print()
    
    # Create solver
    solver = DRDFeasibilitySolver(conjugate_type='exponential')
    
    # Test single option first
    print("1. Single Option Test:")
    single_result = solver.solve_single_option(
        risks_pre_df.iloc[0].values, 
        risks_post_df.iloc[0].values,
        rho, r, alpha
    )
    print(f"   Strategy_0 feasible: {single_result['is_feasible']}")
    print(f"   Constraint values: {single_result['constraint1_value']:.6f}, {single_result['constraint2_value']:.6f}")
    print(f"   Optimal lambda values: {single_result['optimal_lambda1']:.6f}, {single_result['optimal_lambda2']:.6f}")
    print()
    
    # Batch processing with simple loop
    print("2. Batch Processing:")
    import time
    start_time = time.time()
    
    results = solver.solve_batch(risks_pre_df, risks_post_df, rho, r, alpha)
    
    processing_time = time.time() - start_time
    print(f"   Total time: {processing_time:.3f} seconds")
    print(f"   Average time per option: {processing_time/n_options:.4f} seconds")
    print()
    """
    
    def __init__(self, conjugate_type='exponential'):
        """
        Args:
            conjugate_type: Type of conjugate function ('exponential')
        """
        self.conjugate_type = conjugate_type
    
    def phi_star(self, t: np.ndarray) -> np.ndarray:
        """
        Vectorized conjugate function phi*(t).
        
        For exponential case: phi*(t) = t*log(t) - t if t > 0, else 0
        """
        if self.conjugate_type == 'exponential':
            result = np.zeros_like(t)
            positive_mask = t > 0
            t_pos = t[positive_mask]
            result[positive_mask] = t_pos * np.log(t_pos) - t_pos
            return result
        else:
            raise ValueError(f"Unknown conjugate type: {self.conjugate_type}")
    
    def evaluate_constraint(self, lambda_val: float, indicators: np.ndarray, rho: float) -> float:
        """
        Vectorized evaluation:  lambda rho + (1/N) Σᵢ phi*(lambda * indicator_i)
        """
        # Vectorized computation
        lambda_indicators = lambda_val * indicators
        phi_star_values = self.phi_star(lambda_indicators)
        return lambda_val * rho + np.mean(phi_star_values)
    
    def optimize_constraint(self, indicators: np.ndarray, rho: float, lambda_max: float = 100.0) -> Tuple[float, float]:
        """
        Find sup_{lambda>=0} using scipy optimization.
        
        Returns:
            (optimal_lambda, optimal_value)
        """
        # Objective to maximize (minimize negative)
        def objective(lambda_val):
            if lambda_val < 0:
                return 1e10
            return -self.evaluate_constraint(lambda_val, indicators, rho)
        
        # init guess
        best_lambda = 0.0
        best_value = self.evaluate_constraint(0.0, indicators, rho)
        

        try:
            result = optimize.minimize_scalar(
                objective,
                bounds=(1e-8, lambda_max),
                method='bounded'
            )
            
            if result.success:
                value = -result.fun
                if value > best_value:
                    best_value = value
                    best_lambda = result.x
                    
        except Exception:
            print("Optimization failed, using default lambda=0.0")
        
        return best_lambda, best_value
    
    def solve_single_option(self, risks_pre: np.ndarray, risks_post: np.ndarray,
                           rho: float, r: float, alpha: float, lambda_max: float = 100.0) -> dict:
        """
        Solve feasibility for a single option (single row).
        
        Args:
            risks_pre: Pre-treatment risk values f_theta_i(x_pre) for this option
            risks_post: Post-treatment risk values f_theta_i(x_post) for this option  
            rho: Parameter rho
            r: Threshold value
            alpha: Confidence level alpha
            lambda_max: Maximum lambda value
            
        Returns:
            Dictionary with feasibility results
        """
        
        # VECTORIZED: Compute indicators
        indicators_pre = (risks_pre <= r).astype(float)  #  Indicator{f_theta_i(x_pre) <=r}
        indicators_post = (risks_post > r).astype(float)  #  Indicator{f_theta_i(x_post) > r}
        
        # SCIPY OPTIMIZATION: Solve each constraint
        lambda1_opt, sup_val1 = self.optimize_constraint(indicators_pre, rho, lambda_max)
        lambda2_opt, sup_val2 = self.optimize_constraint(indicators_post, rho, lambda_max)
        
        # Check feasibility
        constraint1_satisfied = sup_val1 <= alpha
        constraint2_satisfied = sup_val2 <= alpha
        is_feasible = constraint1_satisfied and constraint2_satisfied
        
        return {
            'is_feasible': is_feasible,
            'constraint1_value': sup_val1,
            'constraint2_value': sup_val2,
            'constraint1_satisfied': constraint1_satisfied,
            'constraint2_satisfied': constraint2_satisfied,
            'optimal_lambda1': lambda1_opt,
            'optimal_lambda2': lambda2_opt,
            'margin1': alpha - sup_val1,
            'margin2': alpha - sup_val2
        }
    
    def solve_batch(self, risks_pre_df: pd.DataFrame, risks_post_df: pd.DataFrame,
                   r: float, alpha: float, lambda_max: float = 100.0, rho : float = 5e-4) -> pd.DataFrame:
        """
        Solve feasibility for multiple options using simple loops.
        
        Args:
            risks_pre_df: DataFrame where each row is an option, each column is a 
                         pre-treatment risk scenario f_theta_i(x_pre)
            risks_post_df: DataFrame where each row is an option, each column is a
                          post-treatment risk scenario f_theta_i(x_post)
            rho: Parameter rho
            r: Threshold value
            alpha: Confidence level alpha  
            lambda_max: Maximum lambda value
            
        Returns:
            DataFrame with results for each option
        """
        
        # Validate inputs
        if risks_pre_df.shape != risks_post_df.shape:
            raise ValueError("Pre and post DataFrames must have same shape")
        
        if risks_pre_df.shape[1] == 0:
            raise ValueError("No risk scenarios provided")
            
        n_options = len(risks_pre_df)
        print(f"Processing {n_options} options...")
        
        # VECTORIZED: Convert to numpy arrays once
        risks_pre_values = risks_pre_df.values  # Shape: (n_options, n_scenarios)
        risks_post_values = risks_post_df.values  # Shape: (n_options, n_scenarios)
        
        # LOOP: Process each option with vectorized operations inside
        results = []
        for i in range(n_options):
            if (i + 1) % 1000 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{n_options} options...")
                
            result = self.solve_single_option(
                risks_pre_values[i], risks_post_values[i],
                rho, r, alpha, lambda_max
            )
            result['option_index'] = i
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.index = risks_pre_df.index
        
        print(f"Completed! Found {results_df['is_feasible'].sum()} feasible options for DRD.")
        return results_df['is_feasible']