import cli
import readfiles as rf

if __name__ == '__main__':
    try:
        rf.ReadFiles()
        # adapter = cli.getAdapter()
        # adapter.run()
    except KeyboardInterrupt:
        print('\n\nInterrupted execution\n')
