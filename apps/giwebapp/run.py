import runpy


def main():
    print('GAL-iD webapp')
    runpy.run_module(
        'giwebapp.app',
        run_name="__main__",
        alter_sys=True
    )


if __name__ == '__main__':
    main()
