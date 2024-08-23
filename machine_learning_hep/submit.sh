#!/bin/bash

##### Configuration

# Analysis stage

STAGE="all"
# STAGE="preproc"
# STAGE="proc_ana"
# STAGE="ana"
# STAGE="variations"
# STAGE="systematics"

# Suffix of the analysis database name

DATABASE="D0Jet_pp"
# DATABASE="LcJet_pp"

# Name of the analysis section in the analysis database

ANALYSIS="jet_obs"

##### Initialisation

DIR_THIS="$(dirname "$(realpath "$0")")"  # This directory
DBDIR="data/data_run3"
DB_DEFAULT="${DIR_THIS}/${DBDIR}/database_ml_parameters_${DATABASE}.yml"
if [[ "${DATABASE}" == "D0Jet_pp" ]]; then
    PREFIX="d0jet_"
elif [[ "${DATABASE}" == "LcJet_pp" ]]; then
    PREFIX="lcjet_"
fi
LOG="log_${STAGE}.log"

##### Execution

echo "$(date) Start"
echo "Running the \"${STAGE}\" stage of the \"${ANALYSIS}\" analysis from the \"${DATABASE}\" database"

if [[ "${STAGE}" == "systematics" ]]; then
    echo "Log file: $LOG"
    python "${DIR_THIS}/analysis/do_systematics.py" -d "${DB_DEFAULT}" -a "${ANALYSIS}" > "${LOG}" 2>&1
elif [[ "${STAGE}" == "variations" ]]; then
    DB_VARIATION="${DIR_THIS}/${DBDIR}/database_variations_${DATABASE}_${ANALYSIS}.yml"
    CONFIG_FILE="${DIR_THIS}/submission/${PREFIX}all.yml"
    "${DIR_THIS}/submit_variations.sh" "${DB_DEFAULT}" "${DB_VARIATION}" "${ANALYSIS}" "${CONFIG_FILE}"
else
    echo "Log file: $LOG"
    CONFIG_FILE="${DIR_THIS}/submission/${PREFIX}${STAGE}.yml"
    CMD_ANA="mlhep -a ${ANALYSIS} -r ${CONFIG_FILE} -d ${DB_DEFAULT} -b --delete"
    ${CMD_ANA} > "${LOG}" 2>&1
fi || echo "Error"

echo -e "\n$(date) Done"
