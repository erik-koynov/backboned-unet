from abc import ABCMeta
import logging
logger = logging.getLogger("models_logger")

class StrictMeta(ABCMeta):
    def __new__(cls,  name, bases, attributes):
        self = super().__new__(cls, name, bases, attributes,)
        logger.info(f"Creating cls, {type(cls), cls, self, type(self)}")
        logger.info(f"Attributes: {attributes}")
        logger.info(f"Metaclass: {self.__abstractmethods__}")
        logger.info(f"MRO: {self.__mro__}")

        for base in self.__mro__[1:]:
            logger.info(f"Base class: {base}")
            if hasattr(base, "__abstractmethods__"):
                for func in base.__abstractmethods__:
                    try:
                        current_func_code = attributes[func].__code__
                    except KeyError:
                        raise TypeError(f"Class must define abstract method {func}")

                    base_func_params = getattr(base, func).__code__.co_varnames
                    logger.info(f"Function {func}'s parameters: {current_func_code.co_varnames[:current_func_code.co_argcount]}")
                    if current_func_code.co_varnames[:current_func_code.co_argcount]!=base_func_params:
                        raise TypeError(f"Class must define {func} with these parameters: {base_func_params}, "
                                        f"but has {current_func_code.co_varnames[:current_func_code.co_argcount]} instead.")


        return self