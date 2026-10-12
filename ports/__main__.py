"""Build, export or check ports through Buildomatic's shared graph workflow."""

import logging

from ports.buildomatic.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("ports.buildomatic.cli").info("Interrupted")
        raise SystemExit(130) from None
    except RuntimeError as error:
        logging.getLogger("ports.buildomatic.cli").error("%s", error)
        raise SystemExit(1) from None
