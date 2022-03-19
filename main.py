import global_quantities
import testing


def main() -> None:

    system = testing.get_testing_system()
    gec = global_quantities.GlobalExponentCalculator(system=system)
    gec.generate_data(dump_file='dump.csv', remove_dump=True, num_points=30)
    print(gec.information)


if __name__ == '__main__':
    main()
