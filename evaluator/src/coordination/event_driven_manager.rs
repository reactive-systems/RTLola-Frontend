use crate::basics::{EvalConfig, EventSource, OutputHandler};
use crate::coordination::{WorkItem, CAP_LOCAL_QUEUE};
use crate::storage::Value;
use crossbeam_channel::Sender;
use std::error::Error;
use std::ops::AddAssign;
use std::sync::Arc;
use std::time::SystemTime;
use streamlab_frontend::ir::LolaIR;

pub(crate) type EventEvaluation = Vec<Value>;

/// Represents the current cycle count for event-driven events.
//TODO(marvin): u128? wouldn't u64 suffice?
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct EventDrivenCycleCount(u128);

type EDM = EventDrivenManager;

impl From<u128> for EventDrivenCycleCount {
    fn from(i: u128) -> EventDrivenCycleCount {
        EventDrivenCycleCount(i)
    }
}

impl AddAssign<u128> for EventDrivenCycleCount {
    fn add_assign(&mut self, i: u128) {
        *self = EventDrivenCycleCount(self.0 + i)
    }
}

pub struct EventDrivenManager {
    current_cycle: EventDrivenCycleCount,
    out_handler: Arc<OutputHandler>,
    event_source: EventSource,
}

impl EventDrivenManager {
    /// Creates a new EventDrivenManager managing event-driven output streams.
    pub(crate) fn setup(ir: LolaIR, config: EvalConfig, out_handler: Arc<OutputHandler>) -> EventDrivenManager {
        let event_source = match EventSource::from(&config.source, &ir) {
            Ok(r) => r,
            Err(e) => panic!("Cannot create input reader: {}", e),
        };

        EDM { current_cycle: 0.into(), out_handler, event_source }
    }

    pub(crate) fn start_online(mut self, work_queue: Sender<WorkItem>) -> ! {
        loop {
            if !self.event_source.has_event() {
                let _ = work_queue.send(WorkItem::End); // Whether it fails or not, we really don't care.
                                                        // Sleep until you slowly fade into nothingness...
                loop {
                    std::thread::sleep(std::time::Duration::new(u64::max_value(), 0))
                }
            }
            let (event, time) = self.event_source.get_event();
            match work_queue.send(WorkItem::Event(event, time)) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending work item. {}", e)),
            }
            self.current_cycle += 1;
        }
    }

    pub(crate) fn start_offline(
        mut self,
        work_queue: Sender<Vec<WorkItem>>,
        time_slot: Sender<SystemTime>,
    ) -> Result<(), Box<dyn Error>> {
        let mut start_time: Option<SystemTime> = None;
        loop {
            let mut local_queue = Vec::with_capacity(CAP_LOCAL_QUEUE);
            for _i in 0..local_queue.capacity() {
                if !self.event_source.has_event() {
                    local_queue.push(WorkItem::End);
                    let _ = work_queue.send(local_queue);
                    return Ok(());
                }
                let (event, time) = self.event_source.get_event();
                if start_time.is_none() {
                    start_time = Some(time);
                    let _ = time_slot.send(time);
                }

                local_queue.push(WorkItem::Event(event, time));
                self.current_cycle += 1;
            }
            match work_queue.send(local_queue) {
                Ok(_) => {}
                Err(e) => self.out_handler.runtime_warning(|| format!("Error when sending local queue. {}", e)),
            }
        }
    }
}