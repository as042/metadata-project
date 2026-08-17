use crate::model::Model;

pub struct Claude;

impl Model for Claude {
    #[inline]
    fn get_response(&self, prompt: String) -> String {
        todo!()
    }
}