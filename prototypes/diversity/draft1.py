class Printer:
    __instance: "Printer" = None
    __initialized: bool = False

    def __init__(self):
        if self.__initialized:
            return
        self.__print_list = []
        self.__initialized = True

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Printer, cls).__new__(cls)
        return cls.__instance

    def __call__(self, *args, **kwargs):
        self.__print_list.append((args, kwargs))

    def print(self):
        for args, kwargs in self.__print_list:
            print(*args, **kwargs)
        self.__print_list.clear()


def main():
    printer = Printer()
    printer("chert")
    printer.print()
    printer2 = Printer()
    printer2.print()

if __name__ == '__main__':
    main()