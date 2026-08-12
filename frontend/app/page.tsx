import { AppShell } from "@/components/layout/AppShell";
import { TranslatorShell } from "@/components/translator/TranslatorShell";

export default function AppHome() {
  return (
    <AppShell>
      <TranslatorShell />
    </AppShell>
  );
}
