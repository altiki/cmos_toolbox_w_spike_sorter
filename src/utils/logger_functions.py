import logging

# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

# Creating an object
console = logging.getLogger('universal')
