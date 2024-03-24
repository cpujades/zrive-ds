class PredictionException(Exception):
    """Base class for prediction-related exceptions"""

    def __init__(self, message):
        self.message = message
        super().__init__(
            message
        )  # Initialize the base Exception class with the message


class UserNotFoundException(Exception):
    """Exception raised when a user ID is not found in the feature store"""

    def __init__(self, message):
        self.message = message
        super().__init__(message)
