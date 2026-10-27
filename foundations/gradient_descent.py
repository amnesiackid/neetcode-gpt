class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = init
        for i in range(iterations):
            x = x - learning_rate * 2 * x
            print(x)
        return round(x, 5)

if __name__ == "__main__":
    solution = Solution()
    x = solution.get_minimizer(10, 0.01,5)
    result = round(x**2, 5)

