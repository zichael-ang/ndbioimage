import os
__all__ = [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(os.path.dirname(__file__))
           if file.endswith('.py') and not file == os.path.basename(__file__)]
