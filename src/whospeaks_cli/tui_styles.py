"""Textual stylesheet kept outside the application state and event code."""

APP_CSS = r"""
Screen {
    background: #0d1117;
    color: #e9eef0;
}

#app-body {
    height: 1fr;
}

#title-bar {
    height: 3;
    padding: 1 2 0 2;
    background: #0d1117;
    color: #ffffff;
}

#app-title {
    width: 1fr;
    height: 1;
    text-style: bold;
}

#app-meta {
    width: auto;
    height: 1;
    color: #8292b4;
}

#status-row, #backend-status-row {
    height: 2;
    padding: 0 2 1 2;
    background: #0d1117;
}

#mode-pill, #readiness-text, .server-state {
    width: auto;
    height: 1;
    margin-right: 1;
    padding: 0 1;
    color: #64f1c4;
    background: #132b2d;
}

#readiness-text {
    color: #ffb845;
    background: #2a2419;
}

#server-status-spacer {
    width: 1fr;
    height: 1;
}

.server-state {
    color: #91a0ad;
    background: #171d20;
}

.server-state.running {
    color: #8dd4a3;
    background: #183527;
}

.server-state.starting {
    color: #78b8ff;
    background: #172b42;
}

.server-state.failed {
    color: #ff929b;
    background: #3b2024;
}

#operation-banner {
    height: 0;
    padding: 0;
    background: #0d1117;
    border: none;
}

#operation-primary, #operation-secondary {
    height: 1;
}

#operation-primary {
    text-style: bold;
    color: #dce6e8;
}

#operation-secondary {
    color: #9faeb4;
}

#operation-banner.status-running {
    background: #0d1117;
    border: none;
}

#operation-banner.status-running #operation-primary {
    color: #91e0d9;
}

#operation-banner.status-success {
    background: #0d1117;
    border: none;
}

#operation-banner.status-success #operation-primary {
    color: #8dd4a3;
}

#operation-banner.status-warning {
    background: #0d1117;
    border: none;
}

#operation-banner.status-warning #operation-primary {
    color: #f2c868;
}

#operation-banner.status-error {
    background: #0d1117;
    border: none;
}

#operation-banner.status-error #operation-primary {
    color: #ff929b;
}

TabbedContent, ContentSwitcher {
    height: 1fr;
}

TabPane {
    padding: 0 1;
    height: 1fr;
}

Tabs {
    height: 3;
    padding: 0 2;
    background: #0d1117;
    color: #8292b4;
    border-bottom: solid #222a32;
}

Tab {
    height: 3;
    padding: 0 2;
    background: #0d1117;
}

Tab.-active {
    color: #ffffff;
    background: #0d1117;
    border-bottom: solid #5798f2;
    text-style: bold;
}

.section-title {
    height: 1;
    color: #ffffff;
    text-style: bold;
}

#setup-options {
    height: 8;
    layout: vertical;
    padding: 0 2;
    background: #0d1117;
}

#target-row, #realtime-row, #translation-install-row, #installer-row {
    width: 1fr;
    height: 2;
    align-vertical: middle;
}

#target-label, #realtime-label, #translation-install-label, #installer-label {
    width: 12;
    height: 1;
    color: #8f9db8;
}

#target-select, #realtime-select {
    width: 1fr;
    height: 2;
    layout: horizontal;
    align-vertical: middle;
    background: transparent;
    border: none;
}

#target-select RadioButton, #realtime-select RadioButton {
    width: auto;
    height: 1;
    padding-right: 1;
}

#quick-language-select {
    width: 28;
    height: 2;
}

#language-label {
    width: 10;
    height: 1;
    color: #8f9db8;
}

#quick-language-select SelectCurrent {
    height: 2;
    border: none;
    background: #1b2227;
}

#live-speakers-checkbox {
    width: auto;
    height: 1;
}

#compatibility-note {
    display: none;
    height: 1;
    padding-left: 12;
    color: #ffb845;
}

Screen.preview-incompatible #setup-options {
    height: 9;
}

Screen.preview-incompatible #compatibility-note {
    display: block;
}

Screen.compact #quick-language-select {
    width: 18;
}

#compact-plan {
    display: none;
    height: 6;
    margin: 1 2 1 2;
    padding: 0 2;
    background: #101824;
    color: #9eb0cd;
    border: solid #27466f;
}

#compact-plan.status-running {
    display: block;
    background: #101824;
    color: #78b8ff;
    border: solid #315d93;
}

#compact-plan.status-success {
    display: block;
    height: 4;
    background: #183527;
    color: #d8f1df;
    border: solid #5eae78;
}

#compact-plan.status-warning {
    display: block;
    height: 4;
    background: #3a3019;
    color: #f8e1a9;
    border: solid #d3a642;
}

#compact-plan.status-error {
    display: block;
    height: 4;
    background: #3b2024;
    color: #ffd0d5;
    border: solid #cc6570;
}

#setup-workspace {
    height: 1fr;
}

#setup-state {
    width: 1fr;
    height: 1fr;
    padding: 0 2;
}

#setup-side {
    width: 39;
    min-width: 34;
    height: 1fr;
    padding: 0 0 0 1;
    margin-left: 1;
    border-left: solid #3d4a50;
}

#plan-summary {
    height: auto;
    max-height: 9;
    margin-bottom: 1;
    padding: 1;
    background: #171d20;
    border: solid #3d4a50;
}

#operation-summary {
    height: auto;
    min-height: 10;
    max-height: 12;
    padding: 1;
    background: #171d20;
    border: solid #3d4a50;
    color: #bdc9cd;
}

#operation-summary.status-running {
    background: #123d40;
    border: solid #51b9b0;
    color: #d7f5f0;
}

#setup-actions, #doctor-actions, #settings-actions, #reports-actions, #translation-actions, #activity-actions {
    dock: bottom;
    height: 4;
    align-horizontal: right;
    padding: 0 2;
    background: #0d1117;
    border-top: solid #222a32;
}

#setup-actions Button {
    width: 1fr;
    min-width: 10;
}

Button {
    min-width: 13;
    height: 3;
    margin-left: 1;
    border: none;
    content-align: center middle;
}

#setup-actions Button:first-of-type {
    margin-left: 0;
}

Button.-primary {
    background: #137d73;
    color: #ffffff;
}

Button.-primary:hover {
    background: #1b9b8e;
}

Button.danger, #cancel-operation {
    background: #9d3f48;
    color: #ffffff;
}

#launch-button {
    background: #3b7a57;
    color: #ffffff;
}

#cancel-operation {
    display: none;
}

DataTable {
    height: 1fr;
    background: #0d1117;
    border: none;
}

#settings-scroll {
    height: 1fr;
}

#settings-grid {
    layout: grid;
    grid-size: 2;
    grid-columns: 1fr 1fr;
    grid-gutter: 1 2;
    height: auto;
    padding: 1 1 4 0;
}

.field {
    height: 5;
}

.field.language-target-field {
    height: 14;
}

#translation-targets-select {
    height: 12;
    border: solid #34424b;
}

.field Label {
    height: 2;
    color: #b8c4ca;
}

Input {
    height: 3;
    background: #1b2227;
    border: solid #46545d;
}

Select {
    height: 3;
    background: transparent;
    border: none;
}

SelectCurrent {
    height: 3;
    color: #e9eef0;
    background: #1b2227;
    border: solid #46545d;
}

SelectCurrent.-has-value Static#label {
    color: #e9eef0;
}

SelectCurrent .arrow {
    color: #65d1c8;
}

Select > SelectOverlay {
    color: #e9eef0;
    background: #1b2227;
    border: solid #46545d;
}

Input:focus, Select:focus > SelectCurrent {
    border: solid #65d1c8;
}

#activity-log {
    height: 1fr;
    margin-top: 1;
    background: #0d1114;
    border: solid #3d4a52;
    padding: 0 1;
}

ConfirmInstallScreen {
    align: center middle;
    background: rgba(5, 8, 10, 0.82);
}

#confirm-dialog {
    width: 76;
    max-width: 92%;
    height: auto;
    max-height: 80%;
    padding: 1 2;
    background: #1b2227;
    border: solid #65d1c8;
}

#confirm-title {
    height: 2;
    text-style: bold;
    color: #ffffff;
}

.confirm-value, .confirm-summary {
    height: auto;
    margin-bottom: 1;
}

#confirm-command {
    height: auto;
    max-height: 6;
    padding: 1;
    background: #0d1114;
    color: #c7d2d8;
    border: solid #3d4a52;
}

.dialog-actions {
    height: 3;
    align-horizontal: right;
    margin-top: 1;
}

Screen.compact #setup-side, Screen.short #setup-side {
    display: none;
}

Screen.short #compact-plan {
    margin: 0 2;
}

Screen.short #compact-plan.status-idle {
    display: block;
    height: 4;
}

Screen.compact #settings-grid {
    grid-size: 1;
    grid-columns: 1fr;
}

Screen.narrow #target-row, Screen.narrow #realtime-row, Screen.narrow #translation-install-row, Screen.narrow #installer-row {
    width: 1fr;
    height: 2;
}

Screen.narrow Tab {
    padding: 0 1;
}
"""
