use std::io::{Read, Write};

use rtlola_reporting::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::RtLolaMir;

#[derive(Serialize, Deserialize)]
struct ExportedMir {
    // we use Value here, so that we can first import the spec and check the frontend_version
    // in case the Mir is different in two versions
    spec: Value,
    hash: Option<String>,
    frontend_version: String,
}

/// Describes an error that can happen when exporting or importing specifications into or from json
#[derive(Debug)]
pub enum ExportError {
    /// An error while parsing or creating json.
    Serde(serde_json::Error),
    /// The frontend version that was used to export does not match the current frontend version
    FrontendVersionMismatch {
        /// the version of the file that was imported
        imported_version: String,
        /// the current frontend version
        current_version: String,
    },
    /// The file that was exported does not have the same hash as the imported one
    HashMismatch {
        /// the hash of the specification that was imported
        imported_hash: String,
        /// the hash of the current specification
        current_hash: String,
    },
}

impl From<ExportError> for Diagnostic {
    fn from(error: ExportError) -> Self {
        match error {
            ExportError::Serde(e) => Diagnostic::error(&format!("Error while exporting to json:\n{e}")),
            ExportError::FrontendVersionMismatch {
                imported_version,
                current_version,
            } => {
                Diagnostic::error(&format!("The imported file was exported with frontend version {imported_version}, but the current frontend version is {current_version}."))
            }
            ExportError::HashMismatch {
                imported_hash,
                current_hash,
            } => Diagnostic::error(&format!("The imported file was exported from a specification with hash {imported_hash}, but the current specification has hash {current_hash}.")),
        }
    }
}

impl RtLolaMir {
    /// Attempts to export the [RtLolaMir] into a json representation.
    /// Json does include the hash of the specification and the frontend version.
    pub fn export<W: Write>(self, spec_str: &str, writer: W) -> Result<(), ExportError> {
        let hash = Sha256::digest(spec_str);
        let hash_str = format!("{hash:X}");
        let frontend_version = env!("CARGO_PKG_VERSION").into();
        let mir_value = serde_json::to_value(self).map_err(ExportError::Serde)?;

        let export = ExportedMir {
            spec: mir_value,
            frontend_version,
            hash: Some(hash_str),
        };

        serde_json::to_writer(writer, &export).map_err(ExportError::Serde)
    }

    /// Attempts to export the [RtLolaMir] into a json representation.
    /// Json does not include the hash of the specification, only the frontend version.
    pub fn export_unchecked<W: Write>(self, writer: W) -> Result<(), ExportError> {
        let frontend_version = env!("CARGO_PKG_VERSION").into();
        let mir_value = serde_json::to_value(self).map_err(ExportError::Serde)?;

        let export = ExportedMir {
            spec: mir_value,
            frontend_version,
            hash: None,
        };

        serde_json::to_writer(writer, &export).map_err(ExportError::Serde)
    }

    /// Attempts to import a json representation of a specification into an [RtLolaMir].
    ///
    /// # Fail
    /// Fails if the frontend version that exported the file does not match the current frontend version.
    /// Fails if the sha256 hash of the file that was exported from does not match the hash of the current file.
    pub fn import<R: Read>(spec_str: &str, reader: R) -> Result<RtLolaMir, ExportError> {
        let ExportedMir {
            spec,
            frontend_version: imported_version,
            hash: imported_hash,
        } = serde_json::from_reader(reader).map_err(ExportError::Serde)?;

        let current_version = env!("CARGO_PKG_VERSION").into();
        if imported_version != current_version {
            return Err(ExportError::FrontendVersionMismatch {
                imported_version,
                current_version,
            });
        }

        if let Some(imported_hash) = imported_hash {
            let current_hash = Sha256::digest(spec_str);
            let current_hash = format!("{current_hash:X}");
            if imported_hash != current_hash {
                return Err(ExportError::HashMismatch {
                    imported_hash,
                    current_hash,
                });
            }
        }

        serde_json::from_value(spec).map_err(ExportError::Serde)
    }

    /// Attempts to import a json representation of a specification into an [RtLolaMir].
    ///
    /// Does not check for correct frontend version or correct hash of the specification.
    pub fn import_unchecked<R: Read>(reader: R) -> Result<RtLolaMir, ExportError> {
        let ExportedMir { spec, .. } = serde_json::from_reader(reader).map_err(ExportError::Serde)?;
        serde_json::from_value(spec).map_err(ExportError::Serde)
    }
}

#[cfg(test)]
mod tests {
    use rtlola_parser::ParserConfig;
    use serde_json::json;

    use super::ExportedMir;
    use crate::export::ExportError;
    use crate::{parse, RtLolaMir};

    #[test]
    fn export_and_import() {
        let spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 5 \"test\"";
        let config = ParserConfig::for_string(spec.into());
        let mir = parse(config).expect("should parse");

        let mut exported = Vec::new();
        mir.export(spec, &mut exported).expect("should be able to export");

        let imported = RtLolaMir::import(spec, &exported[..]).expect("should be able to import");

        let config = ParserConfig::for_string(spec.into());
        let mir = parse(config).expect("should parse");

        assert_eq!(mir, imported);
    }

    #[test]
    fn wrong_checksum() {
        let spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 5 \"test\"";
        let config = ParserConfig::for_string(spec.into());
        let mir = parse(config).expect("should parse");

        let mut exported = Vec::new();
        mir.export(spec, &mut exported).expect("should be able to export");

        let new_spec = "input a : UInt64\ninput b : UInt64\noutput c := a + b\ntrigger c > 10 \"test\"";

        let imported = RtLolaMir::import(new_spec, &exported[..]);

        assert!(matches!(imported, Err(ExportError::HashMismatch { .. })));

        let imported_unchecked = RtLolaMir::import_unchecked(&exported[..]);
        assert!(imported_unchecked.is_ok());
    }

    #[test]
    fn wrong_frontend_version() {
        let spec = "input a : UInt64";
        let mir = parse(ParserConfig::for_string(spec.into())).expect("should parse");

        let mut exported = Vec::new();
        mir.export(spec, &mut exported).expect("should be able to export");

        let mut exported: ExportedMir = serde_json::from_slice(&exported[..]).unwrap();
        exported.frontend_version = "a-really-old-version".into();
        let exported_json = serde_json::to_string(&exported).unwrap();

        let res = RtLolaMir::import(spec, exported_json.as_bytes());

        if let Err(ExportError::FrontendVersionMismatch {
            imported_version,
            current_version,
        }) = res
        {
            assert_eq!(&imported_version, "a-really-old-version");
            assert_eq!(&current_version, env!("CARGO_PKG_VERSION"));
        } else {
            panic!()
        }
    }

    #[test]
    fn wrong_frontend_version2() {
        // here we test if we get the correct error message, even if the imported MIR has a different structure than the current one

        let spec = "input a : UInt64";

        let old_frontend_version = String::from("a-really-old-version");
        let hash = String::from("e0acae2a6dae78313cac1d00a7d5f5fb996289a147c4d86d2b9a24234a445111");
        let old_version = json!({"frontend_version": old_frontend_version, "spec": {"very": ["different", "structure"]}, "hash": hash});
        let old_version = serde_json::to_vec(&old_version).unwrap();

        let res = RtLolaMir::import(spec, &old_version[..]);

        if let Err(ExportError::FrontendVersionMismatch { imported_version, .. }) = res {
            assert_eq!(&imported_version, "a-really-old-version");
        } else {
            panic!()
        }
    }
}
