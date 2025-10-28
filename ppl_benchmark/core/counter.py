class Counter:
    """
    A simple counter class that keeps track of a count value.
    
    This class provides methods to increment, reset, and retrieve the current count.

    #todo: Make this counter thread-safe to support parallel MCMC-chains and parallel SVI particles. 
    """
    
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def reset(self):
        """Reset the counter to 0."""
        self.count = 0
        
    def get(self):
        """
        Get the current count value.
        
        Returns:
            int: The current count value.
        """
        return self.count
