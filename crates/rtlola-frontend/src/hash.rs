//! Allows attaching a hash to an [RtLolaMir]. Used when exporting/importing
//! an [RtLolaMir] with serde::Serialize/Deserialize and check, whether the
//! frontend version or specification did not change or whether the specification
//! has to be parsed again.

use rtlola_parser::ParserConfig;
use rtlola_reporting::Diagnostic;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::RtLolaMir;

/// Represents an [RtLolaMir] with a hash of the specification and frontend version attached to it.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HashedMir {
    // we use Value here, so that we can first import the spec and check the frontend_version
    // in case the Mir is different in two versions
    spec: RtLolaMir,
    /// the hash of the specification and frontend version
    hash: [u8; 32],
}

/// Describes an error that can happen when checking [HashedMir]'s.
#[derive(Debug, Clone, Copy)]
pub enum HashError {
    /// The file that was exported does not have the same hash as the imported one
    HashMismatch {
        /// the hash of the specification that was imported
        imported_hash: [u8; 32],
        /// the hash of the current specification
        current_hash: [u8; 32],
    },
}

impl From<HashError> for Diagnostic {
    fn from(error: HashError) -> Self {
        match error {
            HashError::HashMismatch {
                imported_hash,
                current_hash,
            } => {
                let imported_hash = imported_hash
                    .iter()
                    .map(|byte| format!("{:02X}", byte))
                    .collect::<String>();
                let current_hash = current_hash
                    .iter()
                    .map(|byte| format!("{:02X}", byte))
                    .collect::<String>();
                Diagnostic::error(&format!("The imported file was exported from a specification with hash {imported_hash}, but the current specification has hash {current_hash}."))
            },
        }
    }
}

fn hash_spec(config: &ParserConfig) -> [u8; 32] {
    Sha256::new()
        .chain_update(config.spec())
        .chain_update(env!("CARGO_PKG_VERSION"))
        .finalize()
        .into()
}

impl HashedMir {
    /// Checks the hash of the [HashedMir].
    pub fn check(self, config: &ParserConfig) -> Result<Self, HashError> {
        let Self { spec, hash } = self;
        let current_hash = hash_spec(config);
        if hash != current_hash {
            return Err(HashError::HashMismatch {
                imported_hash: hash,
                current_hash,
            });
        }
        Ok(Self { spec, hash })
    }
}

impl From<HashedMir> for RtLolaMir {
    fn from(value: HashedMir) -> Self {
        value.spec
    }
}

impl RtLolaMir {
    /// Adds a hash of the specification and frontend version to the mir.
    pub fn hash(self, config: &ParserConfig) -> HashedMir {
        let hash = hash_spec(config);
        HashedMir { spec: self, hash }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use rtlola_parser::ParserConfig;

    use super::HashedMir;
    use crate::{parse, RtLolaMir};

    fn to_json(mir: HashedMir) -> Result<Vec<u8>, serde_json::Error> {
        let mut v = Vec::new();
        serde_json::to_writer(&mut v, &mir)?;
        Ok(v)
    }

    fn from_json(v: Vec<u8>) -> Result<HashedMir, serde_json::Error> {
        serde_json::from_reader(&v[..])
    }

    #[test]
    fn export_and_import() {
        let spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 5 \"test\"";
        let config = ParserConfig::for_string(spec.into());
        let mir = parse(&config).expect("should parse");

        let exported_mir = mir.hash(&config);

        let json = to_json(exported_mir).expect("should be able to export");

        let imported: RtLolaMir = from_json(json)
            .expect("should parse")
            .check(&config)
            .expect("should pass")
            .into();

        let mir = parse(&config).expect("should parse");

        assert_eq!(mir, imported);
    }

    #[test]
    fn wrong_checksum() {
        let spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 5 \"test\"";
        let config = ParserConfig::for_string(spec.into());
        let mir = parse(&config).expect("should parse");

        let exported_mir = mir.hash(&config);
        let exported = to_json(exported_mir).unwrap();

        let new_spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 10 \"test\"";
        let new_config = ParserConfig::for_string(new_spec.into());

        let imported = from_json(exported).expect("should parse");

        imported
            .clone()
            .check(&new_config)
            .expect_err("should fail because checksum");

        RtLolaMir::try_from(imported).expect("should work");
    }
}
