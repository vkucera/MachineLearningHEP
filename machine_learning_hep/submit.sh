#!/bin/bash

# CONFIG="default_complete"
# CONFIG="d0jet_all"
# CONFIG="d0jet_ana"
# CONFIG="lcjet_all"
CONFIG="lcjet_ana"
# CONFIG="variations"

DBDIR="data_run3"

DATABASE="LcJet_pp"

ANALYSIS="jet_obs"

DB_DEFAULT="data/${DBDIR}/database_ml_parameters_${DATABASE}.yml"
# DIR_RESULTS="/data/DerivedResultsJets/D0kAnywithJets/vAN-20200304_ROOT6-1/"

if [[ "${CONFIG}" == "variations" ]]; then
    echo "Running the variation script for the ${ANALYSIS} analysis of ${DATABASE}"
    DATABASE_VARIATION="${DATABASE}_${ANALYSIS}"
    DB_VARIATION="data/${DBDIR}/database_variations_${DATABASE_VARIATION}.yml"
    ./submit_variations.sh ${DB_DEFAULT} ${DB_VARIATION} ${ANALYSIS}
else
    CONFIG="submission/${CONFIG}.yml"
    CMD_ANA="mlhep -a ${ANALYSIS} -r ${CONFIG} -d ${DB_DEFAULT} -c"
    echo "Running the ${CONFIG} configuration of the ${ANALYSIS} analysis of ${DATABASE}"
    \time -f "time: %E\nCPU: %P" ${CMD_ANA}
fi

# Exit if error.
if [ ! $? -eq 0 ]; then echo "Error"; exit 1; fi

echo -e "\n$(date)"

# echo -e "\nCleaning ${DIR_RESULTS}"
# ./clean_results.sh ${DIR_RESULTS}

echo -e "\nDone"

exit 0
