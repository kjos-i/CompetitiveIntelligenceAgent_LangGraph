"""Scheduled automation using APScheduler.

Runs the automated lookout on a recurring schedule. Two modes are supported,
configured via SCHEDULE_INTERVAL_MINUTES in config.py:
- Interval mode (SCHEDULE_INTERVAL_MINUTES > 0): runs every N minutes.
- Cron mode (SCHEDULE_INTERVAL_MINUTES = 0): runs at a fixed time of day
  (SCHEDULE_HOUR, SCHEDULE_MINUTE, SCHEDULE_DAYS).

Keep this script running in a terminal window or deploy via a process manager
(systemd, supervisor, etc).

Usage:
    python launch_schedule_runner.py

Press Ctrl+C to stop.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agent_modes import run_automated_lookout
from config import SCHEDULE_DAYS, SCHEDULE_HOUR, SCHEDULE_INTERVAL_MINUTES, SCHEDULE_MINUTE
from utils import setup_logger

logger = setup_logger("scheduler_runner")


async def main() -> None:
    """Run the scheduler indefinitely."""
    scheduler = AsyncIOScheduler()
    
    if SCHEDULE_INTERVAL_MINUTES > 0:
        scheduler.add_job(
            run_automated_lookout,
            'interval',
            minutes=SCHEDULE_INTERVAL_MINUTES,
            next_run_time=datetime.now(),
            id='automated_lookout'
        )
        schedule_desc = f"Every {SCHEDULE_INTERVAL_MINUTES} minute(s)"
    else:
        scheduler.add_job(
            run_automated_lookout,
            'cron',
            hour=SCHEDULE_HOUR,
            minute=SCHEDULE_MINUTE,
            day_of_week=SCHEDULE_DAYS,
            id='automated_lookout'
        )
        schedule_desc = f"Daily {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} ({SCHEDULE_DAYS})"
    
    scheduler.start()
    
    logger.info("=" * 60)
    logger.info("✓ Scheduler started")
    logger.info(f"Automated lookout scheduled: {schedule_desc}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    print(f"\n✓ Scheduler started. Automated lookout will run: {schedule_desc}.")
    print("Press Ctrl+C to stop.\n")
    
    # Keep the script running
    try:
        while True:
            await asyncio.sleep(3600)  # Check every hour
    except (KeyboardInterrupt, SystemExit):
        logger.info("⏹ Scheduler shutdown initiated")
        scheduler.shutdown()
        print("\n✓ Scheduler shut down gracefully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"⚠ Scheduler error: {e}", exc_info=True)
        print(f"\n⚠ Scheduler error: {e}")
        scheduler.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"⚠ Failed to start scheduler: {e}", exc_info=True)
        print(f"⚠ Failed to start scheduler: {e}")
        sys.exit(1)
