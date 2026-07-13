import {createLiveAppContext} from "./live_context.js";
import {installMediaCapture} from "./media_capture.js";
import {installSessionTransport} from "./session_transport.js";
import {installSavedReports} from "./saved_reports.js";
import {installTranscriptTranslation} from "./transcript_translation.js";
import {installTranscriptReview} from "./transcript_review.js";
import {installSpeakerPanel} from "./speaker_panel.js";
import {installTranscriptRender} from "./transcript_render.js";
import {installLiveBindings} from "./live_bindings.js";

export function bootstrapLiveApp() {
  const context = createLiveAppContext();
  installMediaCapture(context);
  installSessionTransport(context);
  installSavedReports(context);
  installTranscriptTranslation(context);
  installTranscriptReview(context);
  installSpeakerPanel(context);
  installTranscriptRender(context);
  installLiveBindings(context);
  for (const activate of context.activators) activate();
  return () => context.appResources.dispose();
}

const disposeLiveApp = bootstrapLiveApp();
export default disposeLiveApp;
