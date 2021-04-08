#!/bin/bash
clearml-data create --project uchicago --name dataset1
clearml-data sync --folder ./data/
