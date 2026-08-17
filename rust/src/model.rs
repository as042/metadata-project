pub mod claude;

pub trait Model {
    fn get_response(&self, prompt: String) -> String;
}